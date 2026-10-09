/* NIHCC 胸部X線画像検索ローカルRAG フロントエンド
 * 依存ライブラリなしの単一ページ実装。API は同一オリジンの /api へ reverse proxy される。
 */
'use strict';

/* 検索の単位は画像 1 枚の胸部野（肺野＋縦隔＋横隔膜）である。部位（RUL など）の
 * 選択や部位ごとの順位は持たない。 */

const state = {
  token: sessionStorage.getItem('token') || null,
  role: sessionStorage.getItem('role') || null,
  user: sessionStorage.getItem('user') || null,
  image: null,
  validationId: null,
  lastSearch: null,
  jobStream: null,
  objectUrls: [],
  policy: { min_length: 8, max_length: 128, reset_token_ttl_minutes: 10 },
  //: 所見ラベルの英語 -> 日本語。/api/v1/searches/support/phrases から受け取る。
  //: **画面に表を持たない。** 表示名と絞り込みの受け口が同じ表から出ないと、
  //: 画面には出ているのに絞り込みでは一致しない語が生まれる。
  findingLabels: {},
  users: [],
  editingUserId: null,
};

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  (Array.isArray(children) ? children : [children]).forEach((child) => {
    if (child == null) return;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  });
  return node;
};

/* Idempotency-Key 用の UUID v4。
 * crypto.randomUUID() は secure context（HTTPS / localhost）でしか公開されないため、
 * IP 指定の平文 HTTP でも動くよう getRandomValues による生成を用意する。 */
const uuid = () => {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;  // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80;  // variant 10
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};

/* ---------------------------------------------------------------- API */

async function api(path, { method = 'GET', body, headers = {}, raw = false } = {}) {
  const options = { method, headers: { ...headers } };
  if (state.token) options.headers.Authorization = `Bearer ${state.token}`;
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (response.status === 401) {
    logout();
    throw new Error('セッションが無効です。再度ログインしてください。');
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const problem = await response.json();
      detail = `${problem.title || 'エラー'}: ${problem.detail || ''}` +
        (problem.code ? ` [${problem.code}]` : '') +
        (problem.errors?.length ? `\n${problem.errors.map((e) => `- ${e.field}: ${e.reason}`).join('\n')}` : '');
    } catch (_) { /* problem+json でない応答 */ }
    throw new Error(detail);
  }
  return raw ? response : response.json();
}

/* 画像は Authorization ヘッダが必要なため fetch して blob URL 化する。
 * Top-K 件ぶんを一斉に要求すると API を飽和させるため、同時実行数を制限し、
 * 表示領域に入ったカードだけを読み込む（IntersectionObserver）。 */
const imageQueue = { active: 0, limit: 4, waiting: [] };

function acquireSlot() {
  if (imageQueue.active < imageQueue.limit) {
    imageQueue.active += 1;
    return Promise.resolve();
  }
  return new Promise((resolve) => imageQueue.waiting.push(resolve));
}

function releaseSlot() {
  const next = imageQueue.waiting.shift();
  if (next) next();
  else imageQueue.active -= 1;
}

async function authedImageUrl(path) {
  await acquireSlot();
  try {
    const response = await api(path, { raw: true });
    const url = URL.createObjectURL(await response.blob());
    state.objectUrls.push(url);
    return url;
  } finally {
    releaseSlot();
  }
}

const lazyObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) loadLazyImage(entry.target);
  });
}, { rootMargin: '300px' });

function loadLazyImage(img) {
  lazyObserver.unobserve(img);
  const path = img.dataset.src;
  if (!path) return;
  delete img.dataset.src;
  authedImageUrl(path)
    .then((url) => { img.src = url; })
    .catch(() => { img.alt = `${img.alt}（読み込みに失敗しました）`; });
}

function lazyImage(path, alt) {
  const img = el('img', { alt, loading: 'lazy' });
  img.dataset.src = path;
  lazyObserver.observe(img);
  return img;
}

/* hidden な要素は IntersectionObserver が交差判定を行わないため、
 * 表示へ切り替えた時点で未読み込み分を明示的に読み込む。 */
function flushLazy(container) {
  container.querySelectorAll('img[data-src]').forEach(loadLazyImage);
}

function revokeImages() {
  state.objectUrls.forEach((url) => URL.revokeObjectURL(url));
  state.objectUrls = [];
}

/* ------------------------------------------------------------- 認証 */

function applySession() {
  const active = Boolean(state.token);
  $('view-login').hidden = active;
  $('view-main').hidden = !active;
  $('logout').hidden = !active;
  $('session-user').textContent = active ? `${state.user}（${state.role}）` : '未ログイン';
  if (active) {
    const operator = state.role === 'operator' || state.role === 'admin';
    const admin = state.role === 'admin';
    $('run-validate').disabled = !operator;
    $('run-ingest').disabled = true;
    document.querySelectorAll('[data-tab="ingest"]').forEach((tab) => { tab.hidden = !operator; });
    // ユーザー管理は admin のみ
    document.querySelectorAll('[data-tab="users"]').forEach((tab) => { tab.hidden = !admin; });
  } else {
    showView('login');
  }
}

/* 画面（ログイン / パスワード忘れ / 再設定 / メイン）の切り替え */
function showView(name) {
  const views = { login: 'view-login', forgot: 'view-forgot', reset: 'view-reset', main: 'view-main' };
  Object.entries(views).forEach(([key, id]) => { $(id).hidden = key !== name; });
}

function notify(id, message, isError = false) {
  const node = $(id);
  if (!node) return;
  node.textContent = message;
  node.hidden = !message;
  if (isError) node.classList.add('error');
}

function applyPolicyHints() {
  const text = `パスワードは ${state.policy.min_length} 文字以上 ${state.policy.max_length} 文字以下です。`;
  document.querySelectorAll('.policy-hint').forEach((node) => { node.textContent = text; });
  const ttl = $('forgot-ttl');
  if (ttl) ttl.textContent = String(state.policy.reset_token_ttl_minutes);
}

async function loadPolicy() {
  try {
    state.policy = await api('/api/v1/auth/password-policy');
  } catch (_) { /* 既定値のまま */ }
  applyPolicyHints();
}

function logout() {
  state.token = state.role = state.user = null;
  sessionStorage.clear();
  applySession();
}

$('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('login-error').hidden = true;
  try {
    const result = await api('/api/v1/auth/token', {
      method: 'POST',
      body: { username: $('login-user').value, password: $('login-password').value },
    });
    state.token = result.access_token;
    state.role = result.role;
    state.user = $('login-user').value;
    sessionStorage.setItem('token', state.token);
    sessionStorage.setItem('role', state.role);
    sessionStorage.setItem('user', state.user);
    applySession();
    showView('main');
    notify('login-error', '');
    loadPhrases();
    loadRecentJobs();
  } catch (error) {
    $('login-error').textContent = error.message;
    $('login-error').hidden = false;
  }
});

$('logout').addEventListener('click', logout);

/* --------------------------------------------------------------- タブ */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    document.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add('active');
    if (tab.dataset.tab === 'admin') loadAdmin();
    if (tab.dataset.tab === 'jobs') loadRecentJobs();
    if (tab.dataset.tab === 'users') loadUsers();
  });
});

/* ------------------------------------------------------------ 検索画面 */

const dropzone = $('dropzone');
dropzone.addEventListener('click', () => $('file-input').click());
['dragenter', 'dragover'].forEach((type) =>
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add('hover'); }));
['dragleave', 'drop'].forEach((type) =>
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove('hover'); }));
dropzone.addEventListener('drop', (e) => setImage(e.dataTransfer.files[0]));
$('file-input').addEventListener('change', (e) => setImage(e.target.files[0]));
$('clear-image').addEventListener('click', () => setImage(null));

function setImage(file) {
  state.image = file || null;
  $('preview-wrap').hidden = !file;
  $('dropzone').hidden = Boolean(file);
  if (!file) return;
  $('preview').src = URL.createObjectURL(file);
  const facts = $('preview-facts');
  facts.innerHTML = '';
  facts.append(el('dt', {}, 'ファイル名'), el('dd', {}, file.name));
  facts.append(el('dt', {}, '形式'), el('dd', {}, file.type || '不明'));
  facts.append(el('dt', {}, 'サイズ'), el('dd', {}, `${(file.size / 1024).toFixed(0)} KiB`));
  const img = new Image();
  img.onload = () => facts.append(el('dt', {}, '解像度'), el('dd', {}, `${img.width}×${img.height}`));
  img.src = $('preview').src;
}

function syncMode() {
  const mode = $('mode').value;
  $('text-block').hidden = mode === 'IMAGE_TO_IMAGE';
  $('weight-block').hidden = mode !== 'HYBRID';
}
$('mode').addEventListener('change', syncMode);
$('image-weight').addEventListener('input', (e) => { $('weight-out').textContent = Number(e.target.value).toFixed(2); });

async function loadPhrases() {
  try {
    const data = await api('/api/v1/searches/support/phrases');
    const box = $('phrases');
    box.innerHTML = '';
    data.phrases.forEach((phrase) => {
      const chip = el('span', { className: 'phrase', title: phrase.en }, phrase.ja);
      chip.addEventListener('click', () => { $('text').value = phrase.ja; });
      box.append(chip);
    });
    (data.findings || []).forEach((finding) => { state.findingLabels[finding.en] = finding.ja; });
    fillFindingOptions(data.findings || []);
  } catch (_) { /* 未ログイン時は無視 */ }
}

//: 所見ラベルを日本語で出す。表に無い値はそのまま出す（新しい所見が増えたとき、
//: 表示が消えるより英語で出る方が良い）。
function findingLabel(label) {
  return state.findingLabels[label] || label;
}

function fillFindingOptions(findings) {
  const list = $('finding-options');
  if (!list) return;
  list.innerHTML = '';
  findings.forEach((finding) => {
    // 日本語を候補に出す。英語も受け付けるので、併記して対応が分かるようにする。
    list.append(el('option', { value: finding.ja }, `${finding.ja}（${finding.en}）`));
  });
}

$('run-search').addEventListener('click', async () => {
  const button = $('run-search');
  $('search-error').hidden = true;
  $('search-status').textContent = '';
  const mode = $('mode').value;

  const request = {
    mode,
    top_k: Number($('top-k').value),
    min_similarity: Number($('min-similarity').value),
    // 2 段検索の 2 段目。off にすると 1 段のみとの精度比較ができる。
    rerank: $('opt-rerank').checked,
    filters: {
      finding_labels_any: splitList($('filter-labels').value),
      view_positions: splitList($('filter-views').value).map((v) => v.toUpperCase()),
      exclude_image_ids: [],
    },
    return: {
      thumbnail: $('opt-thumb').checked,
      roi_overlay: $('opt-overlay').checked,
      include_source_metadata: false,
    },
  };
  if ($('candidate-k').value) request.candidate_k = Number($('candidate-k').value);
  if (mode !== 'IMAGE_TO_IMAGE') {
    request.text = $('text').value;
    request.text_language = $('text-language').value;
  }
  if (mode === 'HYBRID') request.image_weight = Number($('image-weight').value);

  const form = new FormData();
  form.append('request', JSON.stringify(request));
  if (mode !== 'TEXT_TO_IMAGE') {
    if (!state.image) {
      showSearchError('このモードでは検索対象画像が必要です。');
      return;
    }
    form.append('image', state.image, state.image.name);
  }

  button.disabled = true;
  const started = performance.now();
  try {
    const headers = $('opt-async').checked ? { Prefer: 'respond-async' } : {};
    let result = await api('/api/v1/searches', { method: 'POST', body: form, headers });
    if (result.status === 'PENDING' || result.status === 'RUNNING') {
      $('search-status').textContent = `非同期実行中… search_id=${result.search_id}`;
      result = await pollSearch(result.search_id);
    }
    $('search-status').textContent =
      `所要 ${Math.round(performance.now() - started)} ms（サーバ処理 ${result.duration_ms} ms）`;
    renderResults(result);
    document.querySelector('[data-tab="results"]').click();
  } catch (error) {
    showSearchError(error.message);
  } finally {
    button.disabled = false;
  }
});

function showSearchError(message) {
  $('search-error').textContent = message;
  $('search-error').hidden = false;
}

function splitList(value) {
  return value.split(/[,、\s]+/).map((v) => v.trim()).filter(Boolean);
}

async function pollSearch(searchId) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const result = await api(`/api/v1/searches/${searchId}`);
    if (result.status === 'SUCCEEDED') return result;
    if (result.status === 'FAILED') throw new Error('非同期検索が失敗しました。');
  }
  throw new Error('非同期検索がタイムアウトしました。');
}

/* ---------------------------------------------------------- 結果表示 */
/* 検索の単位は画像 1 枚の胸部野（肺野＋縦隔＋横隔膜）であり、1 本の一覧を出す。 */

function confidenceBadge(value) {
  if (value == null) return el('span', { className: 'badge' }, '胸部野の信頼度 n/a');
  const cls = value >= 0.7 ? 'ok' : value >= 0.35 ? 'warn' : 'bad';
  return el('span', { className: `badge ${cls}` }, `胸部野の信頼度 ${value.toFixed(2)}`);
}

//: 胸部野の品質。OK 以外は抽出が怪しい（片肺が見つからない・左右差が大きいなど）。
const QUALITY_LABELS = {
  OK: null,
  ROI_UNCERTAIN: { text: '胸部野の抽出が不確か', className: 'badge warn' },
};

function qualityBadge(status) {
  const label = QUALITY_LABELS[status];
  return label ? el('span', { className: label.className }, label.text) : null;
}

/* 所見の推定（胸部野の分類器の確率）。
 *
 * **診断ではない。** 見出しを常に添え、データセットのラベル（NLP 由来）とは別の
 * 枠に出して、両者が別物だと分かるようにする。0.5 未満は薄く表示する。 */
function estimateBlock(estimates, heading) {
  if (!estimates || !estimates.length) return null;
  const box = el('div', { className: 'estimates' });
  box.append(el('span', { className: 'label-scope estimate-scope', title:
    '胸部野の画像から学習した分類器が出した確率です。診断ではありません。'
    + 'データセットのラベル（放射線科レポートから NLP で導いたもの）とは別物です。' },
  heading || '所見の推定（研究用・診断ではありません）'));
  estimates.forEach((estimate) => {
    const percent = Math.round(estimate.probability * 100);
    box.append(el('span', {
      className: `estimate ${estimate.probability < 0.5 ? 'faint' : ''}`,
      title: `${estimate.finding}: ${estimate.probability.toFixed(3)}`,
    }, `${findingLabel(estimate.finding)} ${percent}%`));
  });
  return box;
}

/* データセットのラベル（画像単位・NLP 由来）。推定とは別の枠に出す。 */
function labelBlock(item) {
  const labels = el('div', { className: 'labels' });
  labels.append(el('span', { className: 'label-scope' }, 'データセットのラベル'));
  if (item.finding_labels.length) {
    // 英語の元の値は title に残す。データセットの値との突き合わせに要る。
    item.finding_labels.forEach((label) =>
      labels.append(el('span', { className: 'label', title: label }, findingLabel(label))));
  } else {
    labels.append(el('span', { className: 'label' }, '(所見なし)'));
  }
  return labels;
}

/* 検索元（アップロードした画像）。候補と同じく胸部野の輪郭を重ねて見せる。 */
function queryChestCard(result) {
  const card = el('div', { className: 'hit query-hit' });
  const imgs = el('div', { className: 'imgs' });
  imgs.append(lazyImage(result.query_chest_url, '検索元画像の胸部野'));
  card.append(imgs);
  const body = el('div', { className: 'body' });
  body.append(el('div', {}, el('span', { className: 'query-label' }, '検索元')));
  body.append(el('div', {}, confidenceBadge(result.query_roi_confidence)));
  const quality = qualityBadge(result.query_quality_status);
  if (quality) body.append(el('div', {}, quality));
  const estimates = estimateBlock(result.query_finding_estimates, result.estimate_label);
  if (estimates) body.append(estimates);
  card.append(body);
  return card;
}

function hitCard(item, { estimateLabel } = {}) {
  const card = el('div', { className: 'hit' });
  const imgs = el('div', { className: 'imgs' });
  // 胸部野の輪郭を重ねた原画像。取得できない場合はサムネイルへ退避する。
  const url = item.chest_url || item.thumbnail_url;
  imgs.append(lazyImage(url, `${item.source_image_name || ''}（胸部野の輪郭）`));
  card.append(imgs);

  const body = el('div', { className: 'body' });
  const head = el('div', {}, [
    el('span', { className: 'rank' }, `#${item.rank}`),
    ` 類似度 ${item.similarity.toFixed(4)} / 距離 ${item.distance.toFixed(4)}`,
  ]);
  // 2 段検索が効いた候補は、順位が所見プロファイルの厳密距離で決まっている。
  // 1 段目の類似度を併記して、どれだけ並べ替わったかを追えるようにする。
  if (item.reranked) {
    head.append(el('span', { className: 'badge' }, '再ランキング済'));
    if (item.coarse_similarity !== null && item.coarse_similarity !== undefined) {
      head.append(el('span', { className: 'muted small' },
        ` 1段目 ${item.coarse_similarity.toFixed(4)}`));
    }
  }
  body.append(head);
  body.append(el('div', { className: 'muted small' },
    `${item.source_image_name || item.image_id.slice(0, 8)} ・ ${item.view_position || '撮影方向不明'} ・ ${item.patient_ref || ''}`));
  const quality = qualityBadge(item.quality_status);
  if (quality) body.append(el('div', {}, quality));

  body.append(labelBlock(item));
  const estimates = estimateBlock(item.finding_estimates, estimateLabel);
  if (estimates) body.append(estimates);
  body.append(el('div', { className: 'muted small' },
    `ラベルの出典: ${item.label_source || '不明'}（データセット付随・画像単位）`));
  card.append(body);
  return card;
}

//: 2 段検索の適用状況。1 段検索のときは何も出さない。
const RERANK_LABELS = {
  APPLIED: { text: '2段検索', className: 'badge' },
  PARTIAL_COVERAGE: { text: '2段検索（一部のみ）', className: 'badge warn' },
  NO_COVERAGE: { text: '2段検索できず（所見プロファイル未登録）', className: 'badge warn' },
  BUDGET_EXHAUSTED: { text: '2段目を省略（timeout 予算切れ）', className: 'badge warn' },
  PROFILE_EXACT: { text: '所見プロファイルで全件検索', className: 'badge' },
};

function rerankBadge(result) {
  const label = RERANK_LABELS[result.rerank_status];
  if (!label) return null;
  const badge = el('span', { className: label.className }, label.text);
  if (result.rerank_status === 'PROFILE_EXACT') {
    badge.title = '全画像の所見プロファイルと厳密に比べた順位（BiomedCLIP の候補絞り込みなし）';
  } else if (result.stage1_count) {
    badge.title = `1 段目 ${result.stage1_count} 件のうち ${result.reranked_count} 件を所見プロファイルで再スコアリング`;
  }
  return badge;
}

//: 胸郭の形ランキングの状態。空リストの理由を利用者に伝え分ける。
const SHAPE_LABELS = {
  AVAILABLE: { text: '形状で比較', className: 'badge' },
  EMPTY: { text: '形状データ未登録', className: 'badge warn' },
  UNAVAILABLE: { text: '形状検索に失敗', className: 'badge bad' },
};

function queryLungCard(url) {
  // **候補と同じ見せ方で左上に置く。** 比べる相手が無いと、何に似ているのかが
  // 判らない。外周つきで並べているので、検索元も外周つきにする。
  const card = el('div', { className: 'hit query-hit' });
  const figure = el('figure', { className: 'lung-figure' }, [
    lazyImage(url, '検索元画像の肺野外周'),
    el('figcaption', { className: 'muted small' }, '黄色の線が比較に使った肺野の外周'),
  ]);
  card.append(figure);
  const body = el('div', { className: 'body' });
  body.append(el('div', {}, el('span', { className: 'query-label' }, '検索元')));
  body.append(el('div', { className: 'muted small' }, 'この形に近い順で右に並びます。'));
  card.append(body);
  return card;
}

function lungCard(item, options) {
  // **外周を描いた画像を主表示にする。** 何を「肺野」として比較したのかが見えないと、
  // 脊柱・心陰影・横隔膜が正しく抜けているかを確かめられない。
  const card = hitCard(item, options);
  if (!item.lung_outline_url) return card;
  const figure = el('figure', { className: 'lung-figure' }, [
    lazyImage(item.lung_outline_url, `${item.source_image_name || ''} の肺野外周`),
    el('figcaption', { className: 'muted small' }, '黄色の線が比較に使った肺野の外周'),
  ]);
  card.querySelector('.imgs').replaceWith(figure);
  return card;
}

function renderResults(result) {
  revokeImages();
  state.lastSearch = result;
  $('results-empty').hidden = true;
  $('results').hidden = false;

  $('result-meta').textContent =
    `search_id=${result.search_id} ・ mode=${result.mode} ・ ${result.results.length} 件`;

  const warnings = $('result-warnings');
  warnings.innerHTML = '';
  (result.warnings || []).forEach((warning) => warnings.append(el('li', {}, warning)));

  const models = $('result-models');
  models.innerHTML = '';
  Object.entries(result.models || {}).forEach(([key, value]) => {
    models.append(el('dt', {}, key), el('dd', {}, value));
  });
  if (result.estimate_model_version) {
    models.append(el('dt', {}, 'finding estimate'), el('dd', {}, result.estimate_model_version));
  }
  if (result.query) {
    models.append(el('dt', {}, 'query text'), el('dd', {}, result.query.text_original || '(画像のみ)'));
    models.append(el('dt', {}, 'embedded text (en)'), el('dd', {}, result.query.text_normalized_en || '-'));
  }
  const options = { estimateLabel: result.estimate_label };

  // 胸部野の類似症例（主表示）
  const main = $('chest-results');
  main.innerHTML = '';
  const block = el('div', { className: 'card region-block' });
  const head = el('div', { className: 'region-head' }, [
    el('h3', {}, '胸部野が似た症例'),
    el('span', { className: 'badge' }, `${result.timing_ms} ms`),
    el('span', { className: 'badge' }, `${result.results.length} 件`),
  ]);
  const rerank = rerankBadge(result);
  if (rerank) head.append(rerank);
  block.append(head);
  block.append(el('p', { className: 'muted small' },
    '肺野・縦隔・横隔膜を含む胸部野（橙の線の内側）どうしを比べた順位です。'
    + '左端が検索元です。'));
  block.append(el('p', { className: 'muted small' },
    '「データセットのラベル」は放射線科レポートから NLP で導いた画像単位のラベルです。'
    + '「所見の推定」は胸部野の画像から学習した分類器の確率で、診断ではありません。'
    + '両者は別物です。'));
  const cards = el('div', { className: 'cards' });
  if (result.query_chest_url) cards.append(queryChestCard(result));
  if (!result.results.length) {
    block.append(cards);
    block.append(el('p', { className: 'muted small' },
      '該当する候補がありません（検索元の胸部野の信頼度不足による抑制を含む）。'));
  } else {
    result.results.forEach((item) => cards.append(hitCard(item, options)));
    block.append(cards);
  }
  main.append(block);

  // 胸郭の形が似た症例。**所見の判別ではない**ので、主表示とは別の枠に出して
  // 役割の違いを明示する。
  const lung = $('lung-results');
  lung.innerHTML = '';
  if (result.lung_shape_status && result.lung_shape_status !== 'NOT_APPLICABLE') {
    const lungBlock = el('div', { className: 'card region-block' });
    const lungHead = el('div', { className: 'region-head' }, [el('h3', {}, '胸郭の形が似た症例')]);
    const label = SHAPE_LABELS[result.lung_shape_status];
    if (label) lungHead.append(el('span', { className: label.className }, label.text));
    if (result.lung_shape_results.length) {
      lungHead.append(el('span', { className: 'badge' }, `${result.lung_shape_results.length} 件`));
    }
    lungBlock.append(lungHead);
    lungBlock.append(el('p', { className: 'muted small' },
      '肺野全体の輪郭・左右の対称性・縦隔の広がりが近い順です（画像単位）。'
      + '所見の判別ではありません。似た体格・似た写り方の症例を並べて見比べるための軸です。'));
    if (!result.lung_shape_results.length) {
      lungBlock.append(el('p', { className: 'muted small' },
        result.lung_shape_status === 'EMPTY'
          ? 'まだ肺野形状ベクトルが登録されていません。'
          : '胸郭の形による候補を取得できませんでした。'));
    } else {
      const lungCards = el('div', { className: 'cards' });
      if (result.query_lung_outline_url) lungCards.append(queryLungCard(result.query_lung_outline_url));
      result.lung_shape_results.forEach((item) => lungCards.append(lungCard(item, options)));
      lungBlock.append(lungCards);
    }
    lung.append(lungBlock);
  }
}

/* ---------------------------------------------------------- 取り込み */

$('run-validate').addEventListener('click', async () => {
  $('ingest-error').hidden = true;
  $('run-ingest').disabled = true;
  try {
    const body = {
      image_root: $('image-root').value,
      csv_path: $('csv-path').value,
      dataset_version: $('dataset-version').value,
      csv_schema_version: $('csv-schema-version').value,
    };
    if ($('sample-limit').value) body.sample_limit = Number($('sample-limit').value);
    const result = await api('/api/v1/ingestions/validate', { method: 'POST', body });
    state.validationId = result.validation_id;
    $('validation-result').hidden = false;

    const counts = $('validation-counts');
    counts.innerHTML = '';
    counts.append(el('dt', {}, 'validation_id'), el('dd', {}, result.validation_id));
    counts.append(el('dt', {}, '判定'), el('dd', {}, result.valid ? '合格' : '不合格'));
    Object.entries(result.counts).forEach(([key, value]) =>
      counts.append(el('dt', {}, key), el('dd', {}, String(value))));

    const issues = $('validation-issues');
    issues.innerHTML = '';
    issues.append(renderIssues('エラー', result.errors));
    issues.append(renderIssues('警告', result.warnings));
    $('run-ingest').disabled = !result.valid;
  } catch (error) {
    $('ingest-error').textContent = error.message;
    $('ingest-error').hidden = false;
  }
});

function renderIssues(title, issues) {
  const wrap = el('div', {});
  wrap.append(el('h3', {}, `${title}（${issues.length} 件、先頭 50 件まで）`));
  if (!issues.length) {
    wrap.append(el('p', { className: 'muted small' }, 'なし'));
    return wrap;
  }
  const table = el('table');
  table.append(el('tr', {}, [el('th', {}, 'コード'), el('th', {}, '対象'), el('th', {}, '詳細')]));
  issues.slice(0, 50).forEach((issue) =>
    table.append(el('tr', {}, [
      el('td', {}, issue.code),
      el('td', {}, issue.source_image_name || '-'),
      el('td', {}, issue.detail || '-'),
    ])));
  wrap.append(table);
  return wrap;
}

$('ingest-goto-jobs').addEventListener('click', () => {
  document.querySelector('[data-tab="jobs"]').click();
});

$('run-ingest').addEventListener('click', async () => {
  $('ingest-error').hidden = true;
  $('run-ingest').disabled = true;
  try {
    const result = await api('/api/v1/ingestions', {
      method: 'POST',
      headers: { 'Idempotency-Key': uuid() },
      body: {
        validation_id: state.validationId,
        on_duplicate: $('on-duplicate').value,
        dry_run: false,
      },
    });
    $('job-id').value = result.job_id;
    // 取り込み画面に留まったまま進捗を出す（履歴タブでも同じ進捗を追える）
    $('ingest-progress').hidden = false;
    $('ingest-progress-label').textContent = '登録しました';
    $('ingest-progress-percent').textContent = '-';
    $('ingest-progress-detail').textContent = `ジョブ ${result.job_id} を登録しました。進捗を取得しています…`;
    $('ingest-bar').className = 'bar';
    $('ingest-bar').style.width = '0%';
    $('ingest-progress-track').classList.add('indeterminate');
    trackJob(result.job_id);
  } catch (error) {
    $('ingest-error').textContent = error.message;
    $('ingest-error').hidden = false;
    $('run-ingest').disabled = false;
  }
});

/* ------------------------------------------------- 取り込みジョブの進捗 */

const JOB_STATUS_LABELS = {
  PENDING: '待機中', RUNNING: '実行中', SUCCEEDED: '完了',
  PARTIAL: '一部失敗', FAILED: '失敗', CANCELLED: '取消',
};
const JOB_BAR_TONE = { SUCCEEDED: 'ok', PARTIAL: 'warn', FAILED: 'err', CANCELLED: 'warn' };
const JOB_FINISHED = ['SUCCEEDED', 'FAILED', 'PARTIAL', 'CANCELLED'];
/* 取消ボタンを出す状態（backend の CANCELLABLE_JOB_STATUSES と一致させる） */
const JOB_CANCELLABLE = ['PENDING', 'RUNNING'];
const PROGRESS_LABELS = {
  total: '総数', succeeded: '成功', failed: '失敗',
  pending: '待機', running: '実行中', quarantined: '隔離', cancelled: '取消',
};

const jobStatusLabel = (status) => JOB_STATUS_LABELS[status] || status;
const num = (value) => Number(value || 0).toLocaleString('ja-JP');

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '-';
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return `${hours} 時間 ${minutes} 分`;
  if (minutes) return `${minutes} 分 ${total % 60} 秒`;
  return `${total} 秒`;
};

/* 取り込み画面と取り込み履歴の両方に同じ進捗を描く（表示中のものだけ更新する） */
function paintProgress(status) {
  const progress = status.progress || {};
  const total = progress.total || 0;
  // failed は隔離（quarantined）を含む。取消した item も「処理を終えた」側に数える
  const done = (progress.succeeded || 0) + (progress.failed || 0) + (progress.cancelled || 0);
  const percent = total ? Math.min(100, (done / total) * 100) : 0;
  const finished = JOB_FINISHED.includes(status.status);

  // 経過時間から処理速度と残り時間を見積もる（CPU 推論では 1 枚あたり約 2 秒）
  const startedAt = Date.parse(status.started_at || status.created_at);
  const endedAt = status.completed_at ? Date.parse(status.completed_at) : Date.now();
  const elapsed = Number.isFinite(startedAt) ? (endedAt - startedAt) / 1000 : NaN;
  const rate = elapsed > 0 && done > 0 ? done / elapsed : 0;
  const remain = Math.max(0, total - done);

  const parts = [`${num(done)} / ${num(total)} 件`, `成功 ${num(progress.succeeded)}`, `失敗 ${num(progress.failed)}`];
  if (progress.quarantined) parts.push(`うち隔離 ${num(progress.quarantined)}`);
  if (progress.cancelled) parts.push(`取消 ${num(progress.cancelled)}`);
  if (!finished) parts.push(`実行中 ${num(progress.running)}`, `残り ${num(remain)}`);
  if (Number.isFinite(elapsed)) parts.push(`経過 ${formatDuration(elapsed)}`);
  if (!finished && rate > 0 && remain > 0) {
    parts.push(`残り約 ${formatDuration(remain / rate)}`, `${rate.toFixed(1)} 件/秒`);
  }
  const detail = parts.join(' ・ ');
  const tone = JOB_BAR_TONE[status.status] || '';

  ['ingest', 'job'].forEach((prefix) => {
    const block = $(`${prefix}-progress`);
    if (!block || block.hidden) return;
    const track = $(`${prefix}-progress-track`);
    const bar = $(`${prefix}-bar`);
    bar.className = tone ? `bar ${tone}` : 'bar';
    bar.style.width = `${percent.toFixed(1)}%`;
    track.classList.toggle('indeterminate', !finished && total === 0);
    track.setAttribute('aria-valuenow', percent.toFixed(0));
    track.setAttribute('aria-valuetext', `${percent.toFixed(0)}% ${done} / ${total} 件`);
    $(`${prefix}-progress-label`).textContent = jobStatusLabel(status.status);
    $(`${prefix}-progress-percent`).textContent = total ? `${percent.toFixed(1)}%` : '-';
    $(`${prefix}-progress-detail`).textContent = detail;
  });
}

/* ------------------------------------------------------- 取り込み履歴 */

$('track-job').addEventListener('click', () => trackJob($('job-id').value.trim()));
$('job-retry').addEventListener('click', async () => {
  const jobId = $('job-id').value.trim();
  const result = await api(`/api/v1/ingestions/${jobId}/retry`, {
    method: 'POST', headers: { 'Idempotency-Key': uuid() },
  });
  $('job-id').value = result.job_id;
  trackJob(result.job_id);
});

function trackJob(jobId) {
  if (!jobId) return;
  if (state.jobStream) state.jobStream.close();
  $('job-progress').hidden = false;
  $('job-errors-link').href = `/api/v1/ingestions/${jobId}/errors`;
  $('job-errors-link').addEventListener('click', async (event) => {
    event.preventDefault();
    const response = await api(`/api/v1/ingestions/${jobId}/errors`, { raw: true });
    const url = URL.createObjectURL(await response.blob());
    const anchor = el('a', { href: url, download: `ingest-errors-${jobId}.csv` });
    anchor.click();
    URL.revokeObjectURL(url);
  }, { once: true });

  // SSE は Authorization ヘッダを付けられないため、ポーリングで進捗を取得する
  const tick = async () => {
    try {
      const status = await api(`/api/v1/ingestions/${jobId}`);
      renderJob(status);
      if (!JOB_FINISHED.includes(status.status)) {
        state.jobTimer = setTimeout(tick, 1500);
      } else {
        $('run-ingest').disabled = false;
        loadRecentJobs();
      }
    } catch (error) {
      $('run-ingest').disabled = false;
      $('job-counts').innerHTML = '';
      $('job-counts').append(el('dt', {}, 'エラー'), el('dd', {}, error.message));
      $('ingest-progress-detail').textContent = `進捗の取得に失敗しました: ${error.message}`;
      $('ingest-progress-track').classList.remove('indeterminate');
    }
  };
  clearTimeout(state.jobTimer);
  tick();
}

function renderJob(status) {
  paintProgress(status);
  const counts = $('job-counts');
  counts.innerHTML = '';
  counts.append(el('dt', {}, 'ジョブID'), el('dd', {}, status.job_id));
  counts.append(el('dt', {}, '状態'), el('dd', {}, jobStatusLabel(status.status)));
  Object.entries(status.progress).forEach(([key, value]) =>
    counts.append(el('dt', {}, PROGRESS_LABELS[key] || key), el('dd', {}, num(value))));
  Object.entries(status.models || {}).forEach(([key, value]) =>
    counts.append(el('dt', {}, `model.${key}`), el('dd', {}, value)));

  const errors = $('job-errors');
  errors.innerHTML = '';
  if (status.error_summary?.length) {
    const table = el('table');
    table.append(el('tr', {}, [el('th', {}, 'エラーコード'), el('th', {}, '件数')]));
    status.error_summary.forEach((row) =>
      table.append(el('tr', {}, [el('td', {}, row.error_code), el('td', {}, String(row.count))])));
    errors.append(table);
  }
}

async function loadRecentJobs() {
  try {
    const status = await api('/api/v1/admin/status');
    const box = $('recent-jobs');
    box.innerHTML = '';
    // 取消は取り込みと同じく operator 以上のみ（API 側でも Operator で検査する）
    const canCancel = state.role === 'operator' || state.role === 'admin';
    const table = el('table');
    table.append(el('tr', {}, ['ジョブID', '状態', '総数', '成功', '失敗', '作成日時', '操作'].map((h) => el('th', {}, h))));
    (status.recent_jobs || []).forEach((job) => {
      const action = el('td');
      if (canCancel && JOB_CANCELLABLE.includes(job.status)) {
        const button = el('button', { type: 'button', className: 'danger small', textContent: '取消' });
        button.addEventListener('click', (event) => {
          event.stopPropagation();  // 行クリック（進捗表示）を発火させない
          cancelJob(job, button);
        });
        action.append(button);
      }
      const row = el('tr', {}, [
        el('td', {}, job.job_id.slice(0, 8)),
        el('td', {}, jobStatusLabel(job.status)),
        el('td', {}, num(job.total_count)),
        el('td', {}, num(job.succeeded_count)),
        el('td', {}, num(job.failed_count)),
        el('td', {}, new Date(job.created_at).toLocaleString('ja-JP')),
        action,
      ]);
      row.style.cursor = 'pointer';
      row.addEventListener('click', () => { $('job-id').value = job.job_id; trackJob(job.job_id); });
      table.append(row);
    });
    box.append(table);
  } catch (_) { /* 未ログイン時は無視 */ }
}

/* 取消の確認 → API → 一覧の更新。
 * 表示中のジョブは進捗のポーリングが CANCELLED を拾い、完了扱いで自動的に止まる。 */
async function cancelJob(job, button) {
  const remaining = Math.max(0, job.total_count - job.succeeded_count - job.failed_count);
  const message = `ジョブ ${job.job_id.slice(0, 8)} を取り消しますか？\n\n`
    + `未処理の約 ${num(remaining)} 件は処理されません。\n`
    + '処理中の画像はそのまま完了します。この操作は元に戻せません。';
  if (!window.confirm(message)) return;
  $('recent-jobs-error').hidden = true;
  button.disabled = true;
  try {
    await api(`/api/v1/ingestions/${job.job_id}/cancel`, { method: 'POST' });
  } catch (error) {
    $('recent-jobs-error').textContent = `取消に失敗しました: ${error.message}`;
    $('recent-jobs-error').hidden = false;
  } finally {
    loadRecentJobs();
  }
}

/* -------------------------------------------------------------- 管理 */

$('refresh-admin').addEventListener('click', loadAdmin);

function fineCoverage(cluster) {
  if (!cluster.available) return '-';
  const total = cluster.row_count ?? 0;
  const fine = cluster.fine_row_count ?? 0;
  if (!total) return '-';
  if (!fine) return '未導入';
  return fine >= total ? `${fine} (全件)` : `${fine} / ${total} 要再取込`;
}

async function loadAdmin() {
  const box = $('admin-content');
  box.innerHTML = '<p class="muted">読み込み中…</p>';
  try {
    const status = await api('/api/v1/admin/status');
    box.innerHTML = '';

    // ベクトル用クラスタは 1 つ（1 Coordinator + 5 Worker）。
    const cluster = status.cluster || {};
    box.append(el('h3', {}, 'ベクトル用クラスタ'));
    const clusterTable = el('table');
    clusterTable.append(el('tr', {}, ['DB', 'Coordinator', '状態', 'Worker（shard 数）', 'shard', '件数', '2段目充填', '形状充填']
      .map((h) => el('th', {}, h))));
    const placement = cluster.shard_placement || {};
    clusterTable.append(el('tr', {}, [
      el('td', {}, cluster.database || '-'),
      el('td', {}, cluster.coordinator || '-'),
      el('td', {}, cluster.available ? '正常' : '停止/未接続'),
      el('td', {}, (cluster.workers || [])
        .map((w) => `${w}（${placement[w.split(':')[0]] ?? '-'}）`).join(' ')),
      el('td', {}, String(cluster.shard_count ?? '-')),
      el('td', {}, String(cluster.row_count ?? '-')),
      // 2 段検索の 2 段目（embedding_fine）を持つ行数。0 は 2 段目未導入（正常）。
      // 0 < 充填数 < 件数 なら再取り込み待ちの取りこぼしがある。
      el('td', {}, fineCoverage(cluster)),
      el('td', {}, String(cluster.lung_row_count ?? '-')),
    ]));
    box.append(clusterTable);

    box.append(el('h3', {}, 'モデル版'));
    const models = el('table');
    models.append(el('tr', {}, ['model_version', '種別', '名称', 'revision', '次元', 'active', '']
      .map((h) => el('th', {}, h))));
    (status.models || []).forEach((model) => {
      const button = el('button', { className: 'ghost' }, '有効化');
      button.addEventListener('click', async () => {
        await api(`/api/v1/admin/models/${encodeURIComponent(model.model_version)}/activate`, { method: 'POST' });
        loadAdmin();
      });
      button.disabled = model.active || state.role !== 'admin';
      models.append(el('tr', {}, [
        el('td', {}, model.model_version),
        el('td', {}, model.model_type),
        el('td', {}, model.model_name),
        el('td', {}, (model.source_revision || '').slice(0, 16)),
        el('td', {}, String(model.embedding_dimension ?? '-')),
        el('td', {}, model.active ? '●' : ''),
        el('td', {}, button),
      ]));
    });
    box.append(models);

    box.append(el('h3', {}, '胸部野 / ベクトル / 所見の推定'));
    const chest = status.chest_assets || {};
    const estimates = status.finding_estimates || {};
    const assets = el('table');
    assets.append(el('tr', {}, ['胸部野 件数', '平均信頼度', '抽出が不確か', '所見の推定 件数', '分類器の版数']
      .map((h) => el('th', {}, h))));
    assets.append(el('tr', {}, [
      el('td', {}, String(chest.count ?? 0)),
      el('td', {}, String(chest.avg_confidence ?? '-')),
      el('td', {}, String(chest.uncertain ?? 0)),
      el('td', {}, String(estimates.count ?? 0)),
      el('td', {}, String(estimates.models ?? 0)),
    ]));
    box.append(assets);

    const vectors = el('table');
    vectors.append(el('tr', {}, ['ベクトル状態', '件数'].map((h) => el('th', {}, h))));
    (status.vector_states || []).forEach((row) =>
      vectors.append(el('tr', {}, [el('td', {}, row.state), el('td', {}, String(row.count))])));
    box.append(vectors);

    box.append(el('h3', {}, 'キュー'));
    const queues = el('table');
    queues.append(el('tr', {}, ['キュー', '滞留数'].map((h) => el('th', {}, h))));
    Object.entries(status.queue_depth || {}).forEach(([queue, depth]) =>
      queues.append(el('tr', {}, [el('td', {}, queue), el('td', {}, depth < 0 ? '取得失敗' : String(depth))])));
    box.append(queues);

    box.append(el('p', { className: 'muted small' },
      `登録画像 ${status.image_count} 件 ・ preprocess ${status.preprocess_version}`));
  } catch (error) {
    box.innerHTML = '';
    box.append(el('p', { className: 'error' }, error.message));
  }
}

/* -------------------------------------------------------------- 初期化 */

/* ============================================================ パスワード再設定 */

$('go-forgot').addEventListener('click', (event) => {
  event.preventDefault();
  notify('forgot-error', '');
  notify('forgot-notice', '');
  showView('forgot');
});

$('back-to-login').addEventListener('click', (event) => {
  event.preventDefault();
  showView('login');
});

$('reset-to-login').addEventListener('click', (event) => {
  event.preventDefault();
  clearResetToken();
  showView('login');
});

$('forgot-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  notify('forgot-error', '');
  notify('forgot-notice', '');
  const button = event.target.querySelector('button[type=submit]');
  button.disabled = true;
  try {
    const result = await api('/api/v1/auth/password-reset/request', {
      method: 'POST',
      body: { email: $('forgot-email').value.trim() },
    });
    // 登録の有無を画面から判別させないため、常に同じ案内を表示する
    notify('forgot-notice', result.detail);
    $('forgot-email').value = '';
  } catch (error) {
    notify('forgot-error', error.message, true);
  } finally {
    button.disabled = false;
  }
});

function resetTokenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  if (!token) return null;
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(token);
  return isUuid ? token : null;
}

function clearResetToken() {
  // token を履歴・ブックマークへ残さない
  window.history.replaceState({}, '', window.location.pathname.replace(/\/reset$/, '/'));
}

$('reset-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  notify('reset-error', '');
  notify('reset-notice', '');
  const token = resetTokenFromUrl();
  if (!token) {
    notify('reset-error', 'URL が不正です。パスワード再設定をやり直してください。', true);
    return;
  }
  const password = $('reset-password').value;
  const confirmation = $('reset-password-confirm').value;
  if (password !== confirmation) {
    notify('reset-error', '確認用パスワードが一致しません。', true);
    return;
  }
  const button = event.target.querySelector('button[type=submit]');
  button.disabled = true;
  try {
    const result = await api('/api/v1/auth/password-reset/confirm', {
      method: 'POST',
      body: { token, new_password: password, new_password_confirmation: confirmation },
    });
    $('reset-password').value = '';
    $('reset-password-confirm').value = '';
    clearResetToken();
    showView('login');
    notify('login-notice', result.detail);
  } catch (error) {
    notify('reset-error', error.message, true);
  } finally {
    button.disabled = false;
  }
});

/* ========================================================== パスワード変更 */

$('password-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  notify('password-error', '');
  notify('password-notice', '');
  const current = $('current-password').value;
  const next = $('new-password').value;
  const confirmation = $('new-password-confirm').value;
  if (next !== confirmation) {
    notify('password-error', '確認用パスワードが一致しません。', true);
    return;
  }
  const button = event.target.querySelector('button[type=submit]');
  button.disabled = true;
  try {
    const result = await api('/api/v1/account/password', {
      method: 'POST',
      body: {
        current_password: current,
        new_password: next,
        new_password_confirmation: confirmation,
      },
    });
    ['current-password', 'new-password', 'new-password-confirm'].forEach((id) => { $(id).value = ''; });
    notify('password-notice', result.detail);
    // パスワード変更で既存トークンは失効するため、数秒後にログイン画面へ戻す
    setTimeout(() => { logout(); notify('login-notice', result.detail); }, 3000);
  } catch (error) {
    notify('password-error', error.message, true);
  } finally {
    button.disabled = false;
  }
});

/* ============================================================ ユーザー管理 */

const ROLE_LABELS = {
  viewer: 'viewer（参照のみ）',
  operator: 'operator（取り込み実行）',
  admin: 'admin（ユーザー管理・モデル切替）',
};

function formatDateTime(value) {
  return value ? new Date(value).toLocaleString('ja-JP') : '-';
}

async function loadUsers() {
  notify('users-error', '');
  const box = $('users-table');
  box.innerHTML = '<p class="muted">読み込み中…</p>';
  try {
    const includeInactive = $('users-include-inactive').checked;
    const data = await api(`/api/v1/users?include_inactive=${includeInactive}`);
    state.users = data.users;
    box.innerHTML = '';

    const table = el('table');
    table.append(el('tr', {}, ['アカウント名', '権限', 'メールアドレス', '状態', '最終ログイン', '']
      .map((h) => el('th', {}, h))));
    data.users.forEach((user) => {
      const detail = el('button', { className: 'ghost' }, '詳細');
      detail.addEventListener('click', () => showUserDetail(user.user_id));
      table.append(el('tr', {}, [
        el('td', {}, user.username),
        el('td', {}, user.role),
        el('td', {}, user.email),
        el('td', {}, user.is_active ? '有効' : '無効'),
        el('td', {}, formatDateTime(user.last_login_at)),
        el('td', {}, detail),
      ]));
    });
    box.append(table);
    box.append(el('p', { className: 'muted small' }, `${data.total} 件`));
  } catch (error) {
    box.innerHTML = '';
    notify('users-error', error.message, true);
  }
}

$('refresh-users').addEventListener('click', loadUsers);
$('users-include-inactive').addEventListener('change', loadUsers);

function openUserForm(user) {
  state.editingUserId = user ? user.user_id : null;
  $('user-form-card').hidden = false;
  $('user-detail-card').hidden = true;
  $('user-form-title').textContent = user ? `ユーザーの編集: ${user.username}` : 'ユーザーの新規追加';
  $('user-username').value = user ? user.username : '';
  // アカウント名は識別子のため編集不可とする
  $('user-username').disabled = Boolean(user);
  $('user-role').value = user ? user.role : 'viewer';
  $('user-email').value = user ? user.email : '';
  $('user-memo').value = user && user.memo ? user.memo : '';
  $('user-active').checked = user ? user.is_active : true;
  $('user-password').value = '';
  $('user-password-confirm').value = '';
  $('user-password').required = !user;
  $('user-password-confirm').required = !user;
  $('user-password-required').hidden = Boolean(user);
  $('user-password-hint').hidden = !user;
  notify('user-form-error', '');
  $('user-form-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

$('new-user').addEventListener('click', () => openUserForm(null));
$('user-cancel').addEventListener('click', () => { $('user-form-card').hidden = true; });

$('user-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  notify('user-form-error', '');
  notify('users-notice', '');
  const password = $('user-password').value;
  const confirmation = $('user-password-confirm').value;
  if (password || confirmation) {
    if (password !== confirmation) {
      notify('user-form-error', '確認用パスワードが一致しません。', true);
      return;
    }
    if (password.length < state.policy.min_length || password.length > state.policy.max_length) {
      notify('user-form-error',
        `パスワードは ${state.policy.min_length} 文字以上 ${state.policy.max_length} 文字以下で入力してください。`, true);
      return;
    }
  }

  const button = $('user-save');
  button.disabled = true;
  try {
    if (state.editingUserId) {
      const body = {
        role: $('user-role').value,
        email: $('user-email').value.trim(),
        memo: $('user-memo').value.trim(),
        is_active: $('user-active').checked,
      };
      if (password) {
        body.password = password;
        body.password_confirmation = confirmation;
      }
      await api(`/api/v1/users/${state.editingUserId}`, { method: 'PATCH', body });
      notify('users-notice', 'ユーザー情報を更新しました。');
    } else {
      await api('/api/v1/users', {
        method: 'POST',
        body: {
          username: $('user-username').value.trim(),
          password,
          password_confirmation: confirmation,
          role: $('user-role').value,
          email: $('user-email').value.trim(),
          memo: $('user-memo').value.trim() || null,
          is_active: $('user-active').checked,
        },
      });
      notify('users-notice', 'ユーザーを追加しました。');
    }
    $('user-form-card').hidden = true;
    loadUsers();
  } catch (error) {
    notify('user-form-error', error.message, true);
  } finally {
    button.disabled = false;
  }
});

async function showUserDetail(userId) {
  notify('users-error', '');
  try {
    const user = await api(`/api/v1/users/${userId}`);
    state.editingUserId = userId;
    $('user-detail-card').hidden = false;
    $('user-form-card').hidden = true;
    const box = $('user-detail');
    box.innerHTML = '';
    const rows = [
      ['アカウント名', user.username],
      ['権限', ROLE_LABELS[user.role] || user.role],
      ['メールアドレス', user.email],
      ['状態', user.is_active ? '有効' : '無効'],
      ['メモ', user.memo || '(なし)'],
      ['最終ログイン', formatDateTime(user.last_login_at)],
      ['パスワード更新', formatDateTime(user.password_changed_at)],
      ['作成', `${formatDateTime(user.created_at)}（${user.created_by || '-'}）`],
      ['更新', `${formatDateTime(user.updated_at)}（${user.updated_by || '-'}）`],
      ['user_id', user.user_id],
    ];
    rows.forEach(([key, value]) => box.append(el('dt', {}, key), el('dd', {}, value)));
    $('user-detail-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (error) {
    notify('users-error', error.message, true);
  }
}

$('user-detail-close').addEventListener('click', () => { $('user-detail-card').hidden = true; });

$('user-detail-edit').addEventListener('click', async () => {
  const user = await api(`/api/v1/users/${state.editingUserId}`);
  openUserForm(user);
});

$('user-detail-delete').addEventListener('click', async () => {
  const user = state.users.find((item) => item.user_id === state.editingUserId);
  const name = user ? user.username : state.editingUserId;
  if (!window.confirm(`ユーザー「${name}」を削除します。取り消せません。よろしいですか？`)) return;
  notify('users-error', '');
  try {
    await api(`/api/v1/users/${state.editingUserId}`, { method: 'DELETE' });
    $('user-detail-card').hidden = true;
    notify('users-notice', `ユーザー「${name}」を削除しました。`);
    loadUsers();
  } catch (error) {
    notify('users-error', error.message, true);
  }
});

/* -------------------------------------------------------------- 初期化 */

syncMode();
applyPolicyHints();
loadPolicy();

const initialResetToken = resetTokenFromUrl();
if (initialResetToken) {
  // 仮パスワード変更 URL でのアクセス。ログイン状態に関わらず再設定画面を表示する。
  showView('reset');
} else {
  applySession();
  if (state.token) {
    showView('main');
    loadPhrases();
    loadRecentJobs();
  }
}
