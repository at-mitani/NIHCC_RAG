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
  //: 定型所見と所見ラベル（英語）。/api/v1/searches/support/phrases から受け取る
  phrases: [],
  findings: [],
  users: [],
  editingUserId: null,
  //: 言語を切り替えたときに描き直すため、最後に描いた応答を持っておく
  lastValidation: null,
  lastJobStatus: null,
  userDetail: null,
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

/* ------------------------------------------------------------ 多言語化 */
/* 表示用の文言は static/i18n/{ja,en,zh}.json に置き、ここには書かない。
 * HTML は data-i18n 系の属性でキーを指し、JS は t('key', { name: value }) で引く。
 * 選んだ言語の辞書に無いキーは日本語へ、日本語にも無ければキーそのものへ倒す
 * （訳し漏れで画面が空になるより、日本語で出る方が気づける）。 */

const LANGUAGES = ['ja', 'en', 'zh'];
//: 日付・数値の書式と <html lang>
const LOCALES = { ja: 'ja-JP', en: 'en-US', zh: 'zh-CN' };
const i18n = { lang: 'ja', dict: {}, fallback: {} };

function t(key, params = {}) {
  const template = i18n.dict[key] ?? i18n.fallback[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

const locale = () => LOCALES[i18n.lang];

//: 辞書に key が無ければ fallback を返す（サーバの値・英語の原文をそのまま出す場合）
function translateOr(key, fallback, params = {}) {
  const text = t(key, params);
  return text === key ? fallback : text;
}

/* サーバが返す文言を表示言語に訳す。サーバは英語の原文に、訳すための id
 * （エラーは message_id、警告は warning_details[].id）と差し込む値 params を添える。
 * 辞書（server.<id>）に無い id は英語の原文をそのまま出す。 */
function serverText(id, params, english) {
  return id ? translateOr(`server.${id}`, english, params || {}) : english;
}

//: 受け付け応答（AcceptedResponse）の文言
const acceptedText = (result) => serverText(result.message_id, result.params, result.detail);

async function loadDictionary(lang) {
  const response = await fetch(`/i18n/${lang}.json`, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`dictionary ${lang}: ${response.status}`);
  return response.json();
}

//: 保存した選択 → ブラウザの言語 → 日本語 の順で決める
function initialLanguage() {
  try {
    const saved = localStorage.getItem('lang');
    if (LANGUAGES.includes(saved)) return saved;
  } catch (_) { /* storage を使えない環境 */ }
  const browser = (navigator.language || '').toLowerCase();
  if (browser.startsWith('zh')) return 'zh';
  if (browser.startsWith('en')) return 'en';
  return 'ja';
}

const I18N_ATTRIBUTES = ['placeholder', 'title', 'alt', 'aria-label'];

function applyStaticTexts(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
  I18N_ATTRIBUTES.forEach((attribute) => {
    root.querySelectorAll(`[data-i18n-${attribute}]`).forEach((node) => {
      node.setAttribute(attribute, t(node.getAttribute(`data-i18n-${attribute}`)));
    });
  });
  document.title = t('app.title');
  document.documentElement.lang = locale();
}

async function setLanguage(lang) {
  if (!LANGUAGES.includes(lang)) lang = 'ja';
  try {
    i18n.dict = lang === 'ja' ? i18n.fallback : await loadDictionary(lang);
  } catch (_) {
    // 辞書を読めなければ日本語で出す（画面を空にしない）
    lang = 'ja';
    i18n.dict = i18n.fallback;
  }
  i18n.lang = lang;
  try { localStorage.setItem('lang', lang); } catch (_) { /* 保存できなくても表示は切り替える */ }
  $('lang-select').value = lang;
  applyStaticTexts();
}

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
  // トークンを付けた要求の 401 はセッション切れ。付けていない要求（ログイン）の 401 は
  // 認証の失敗なので、サーバの理由（ユーザー名またはパスワード違いなど）をそのまま出す
  if (response.status === 401 && state.token) {
    logout();
    throw new Error(t('error.sessionInvalid'));
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const problem = await response.json();
      const message = serverText(problem.message_id || problem.code, problem.params, problem.detail);
      detail = (message || `${problem.title || t('common.error')}`) +
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
    .catch(() => { img.alt = t('error.imageLoadFailed', { alt: img.alt }); });
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
  renderSessionUser();
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

function renderSessionUser() {
  $('session-user').textContent = state.token
    ? t('header.sessionUser', { user: state.user, role: state.role })
    : t('header.notLoggedIn');
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
  const text = t('common.policyHint', { min: state.policy.min_length, max: state.policy.max_length });
  document.querySelectorAll('.policy-hint').forEach((node) => { node.textContent = text; });
  $('forgot-description').textContent = t('forgot.description', { ttl: state.policy.reset_token_ttl_minutes });
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
  renderPreviewFacts();
}

function renderPreviewFacts() {
  const file = state.image;
  if (!file) return;
  const facts = $('preview-facts');
  facts.innerHTML = '';
  facts.append(el('dt', {}, t('search.preview.fileName')), el('dd', {}, file.name));
  facts.append(el('dt', {}, t('search.preview.format')), el('dd', {}, file.type || t('common.unknown')));
  facts.append(el('dt', {}, t('search.preview.size')), el('dd', {}, `${(file.size / 1024).toFixed(0)} KiB`));
  const img = new Image();
  img.onload = () => facts.append(el('dt', {}, t('search.preview.resolution')), el('dd', {}, `${img.width}×${img.height}`));
  img.src = $('preview').src;
}

function syncMode() {
  const mode = $('mode').value;
  $('text-block').hidden = mode === 'IMAGE_TO_IMAGE';
  $('weight-block').hidden = mode !== 'HYBRID';
}
$('mode').addEventListener('change', syncMode);
$('image-weight').addEventListener('input', (e) => { $('weight-out').textContent = Number(e.target.value).toFixed(2); });

/* 定型所見と所見ラベルはサーバから英語だけを受け取り、表示名は辞書
 * （phrase.<英語> / finding.<ラベル>）で表示言語に訳す。
 * 日本語の表示名は、サーバが日本語入力を英語化する表と同じ語にしてある
 * （画面に出ている語を入力すれば解決できる。test_frontend_i18n が検査する）。 */
async function loadPhrases() {
  try {
    const data = await api('/api/v1/searches/support/phrases');
    state.phrases = data.phrases || [];
    state.findings = data.findings || [];
    renderPhrases();
    fillFindingOptions();
  } catch (_) { /* 未ログイン時は無視 */ }
}

//: サーバから英語で受け取った値の表示名。選んだ言語の辞書だけを見る
//: （日本語への退避はしない。英語表示で日本語が出てしまうため）。無ければ英語のまま
const valueLabel = (key, english) => i18n.dict[key] ?? english;
const phraseLabel = (english) => valueLabel(`phrase.${english}`, english);

function renderPhrases() {
  const box = $('phrases');
  box.innerHTML = '';
  state.phrases.forEach((phrase) => {
    const label = phraseLabel(phrase.en);
    const chip = el('span', { className: 'phrase', title: phrase.en }, label);
    chip.addEventListener('click', () => {
      // サーバが英語化できるのは日本語と英語だけ。日本語表示では日本語の語を、
      // それ以外では英語の原文を入れ、入力言語もそれに合わせる。
      const japanese = i18n.lang === 'ja' && label !== phrase.en;
      $('text').value = japanese ? label : phrase.en;
      $('text-language').value = japanese ? 'ja' : 'en';
    });
    box.append(chip);
  });
}

//: 所見ラベルを表示言語で出す。辞書に無い値はそのまま出す（新しい所見が増えたとき、
//: 表示が消えるより英語で出る方が良い）。
const findingLabel = (label) => valueLabel(`finding.${label}`, label);

function fillFindingOptions() {
  const list = $('finding-options');
  if (!list) return;
  list.innerHTML = '';
  state.findings.forEach((finding) => {
    // 絞り込みは日本語と英語の名前しか受け付けない（サーバ側の解決表）。
    // 日本語表示では日本語を、それ以外では英語を候補の値にし、表示名を併記する。
    const name = findingLabel(finding.en);
    const value = i18n.lang === 'ja' ? name : finding.en;
    list.append(el('option', { value }, name === finding.en ? name : `${name}（${finding.en}）`));
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
  if (mode !== 'IMAGE_TO_IMAGE') {
    request.text = $('text').value;
    request.text_language = $('text-language').value;
  }
  if (mode === 'HYBRID') request.image_weight = Number($('image-weight').value);

  const form = new FormData();
  form.append('request', JSON.stringify(request));
  if (mode !== 'TEXT_TO_IMAGE') {
    if (!state.image) {
      showSearchError(t('search.needImage'));
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
      $('search-status').textContent = t('search.asyncRunning', { id: result.search_id });
      result = await pollSearch(result.search_id);
    }
    $('search-status').textContent = t('search.elapsed', {
      client: Math.round(performance.now() - started),
      server: result.duration_ms,
    });
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
    if (result.status === 'FAILED') throw new Error(t('search.asyncFailed'));
  }
  throw new Error(t('search.asyncTimeout'));
}

/* ---------------------------------------------------------- 結果表示 */
/* 検索の単位は画像 1 枚の胸部野（肺野＋縦隔＋横隔膜）であり、1 本の一覧を出す。 */

function confidenceBadge(value) {
  if (value == null) return el('span', { className: 'badge' }, t('results.confidence', { value: 'n/a' }));
  const cls = value >= 0.7 ? 'ok' : value >= 0.35 ? 'warn' : 'bad';
  return el('span', { className: `badge ${cls}` }, t('results.confidence', { value: value.toFixed(2) }));
}

//: 胸部野の品質。OK 以外は抽出が怪しい（片肺が見つからない・左右差が大きいなど）。
//: 文言は辞書の results.quality.<状態>。
const QUALITY_CLASSES = {
  OK: null,
  ROI_UNCERTAIN: 'badge warn',
};

function qualityBadge(status) {
  const className = QUALITY_CLASSES[status];
  return className ? el('span', { className }, t(`results.quality.${status}`)) : null;
}

/* 所見の推定（胸部野の分類器の確率）。
 *
 * **診断ではない。** 見出しを常に添え、データセットのラベル（NLP 由来）とは別の
 * 枠に出して、両者が別物だと分かるようにする。0.5 未満は薄く表示する。
 *
 * 見出しは辞書の results.estimateHeading を表示言語で出す（応答の estimate_label は
 * その英語版）。どの言語の見出しも「診断ではない」ことを含める
 * （test_frontend_i18n で固定している）。 */
function estimateBlock(estimates) {
  if (!estimates || !estimates.length) return null;
  const box = el('div', { className: 'estimates' });
  box.append(el('span', { className: 'label-scope estimate-scope', title: t('results.estimateTitle') },
    t('results.estimateHeading')));
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
  labels.append(el('span', { className: 'label-scope' }, t('results.datasetLabels')));
  if (item.finding_labels.length) {
    // 英語の元の値は title に残す。データセットの値との突き合わせに要る。
    item.finding_labels.forEach((label) =>
      labels.append(el('span', { className: 'label', title: label }, findingLabel(label))));
  } else {
    labels.append(el('span', { className: 'label' }, t('results.noFinding')));
  }
  return labels;
}

/* 検索元（アップロードした画像）。候補と同じく胸部野の輪郭を重ねて見せる。 */
function queryChestCard(result) {
  const card = el('div', { className: 'hit query-hit' });
  const imgs = el('div', { className: 'imgs' });
  imgs.append(lazyImage(result.query_chest_url, t('results.queryChestAlt')));
  card.append(imgs);
  const body = el('div', { className: 'body' });
  body.append(el('div', {}, el('span', { className: 'query-label' }, t('results.query'))));
  body.append(el('div', {}, confidenceBadge(result.query_roi_confidence)));
  const quality = qualityBadge(result.query_quality_status);
  if (quality) body.append(el('div', {}, quality));
  const estimates = estimateBlock(result.query_finding_estimates);
  if (estimates) body.append(estimates);
  card.append(body);
  return card;
}

function hitCard(item) {
  const card = el('div', { className: 'hit' });
  const imgs = el('div', { className: 'imgs' });
  // 胸部野の輪郭を重ねた原画像。取得できない場合はサムネイルへ退避する。
  const url = item.chest_url || item.thumbnail_url;
  imgs.append(lazyImage(url, t('results.chestOutlineAlt', { name: item.source_image_name || '' })));
  card.append(imgs);

  const body = el('div', { className: 'body' });
  const head = el('div', {}, [
    el('span', { className: 'rank' }, `#${item.rank}`),
    t('results.score', { similarity: item.similarity.toFixed(4), distance: item.distance.toFixed(4) }),
  ]);
  // 2 段検索が効いた候補は、順位が所見プロファイルの厳密距離で決まっている。
  // 1 段目の類似度を併記して、どれだけ並べ替わったかを追えるようにする。
  if (item.reranked) {
    head.append(el('span', { className: 'badge' }, t('results.reranked')));
    if (item.coarse_similarity !== null && item.coarse_similarity !== undefined) {
      head.append(el('span', { className: 'muted small' },
        t('results.stage1', { value: item.coarse_similarity.toFixed(4) })));
    }
  }
  body.append(head);
  body.append(el('div', { className: 'muted small' },
    `${item.source_image_name || item.image_id.slice(0, 8)} ・ ${item.view_position || t('results.viewUnknown')} ・ ${item.patient_ref || ''}`));
  const quality = qualityBadge(item.quality_status);
  if (quality) body.append(el('div', {}, quality));

  body.append(labelBlock(item));
  const estimates = estimateBlock(item.finding_estimates);
  if (estimates) body.append(estimates);
  body.append(el('div', { className: 'muted small' },
    t('results.labelSource', { source: item.label_source || t('common.unknown') })));
  card.append(body);
  return card;
}

//: 2 段検索の適用状況。1 段検索のときは何も出さない。文言は辞書の results.rerank.<状態>。
const RERANK_CLASSES = {
  APPLIED: 'badge',
  PARTIAL_COVERAGE: 'badge warn',
  NO_COVERAGE: 'badge warn',
  BUDGET_EXHAUSTED: 'badge warn',
  PROFILE_EXACT: 'badge',
};

function rerankBadge(result) {
  const className = RERANK_CLASSES[result.rerank_status];
  if (!className) return null;
  const badge = el('span', { className }, t(`results.rerank.${result.rerank_status}`));
  if (result.rerank_status === 'PROFILE_EXACT') {
    badge.title = t('results.rerank.profileTitle');
  } else if (result.stage1_count) {
    badge.title = t('results.rerank.stage1Title', {
      stage1: result.stage1_count, reranked: result.reranked_count,
    });
  }
  return badge;
}

//: 胸郭の形ランキングの状態。空リストの理由を利用者に伝え分ける。
//: 文言は辞書の results.shape.<状態>。
const SHAPE_CLASSES = {
  AVAILABLE: 'badge',
  EMPTY: 'badge warn',
  UNAVAILABLE: 'badge bad',
};

function queryLungCard(url) {
  // **候補と同じ見せ方で左上に置く。** 比べる相手が無いと、何に似ているのかが
  // 判らない。外周つきで並べているので、検索元も外周つきにする。
  const card = el('div', { className: 'hit query-hit' });
  const figure = el('figure', { className: 'lung-figure' }, [
    lazyImage(url, t('results.queryLungAlt')),
    el('figcaption', { className: 'muted small' }, t('results.lungCaption')),
  ]);
  card.append(figure);
  const body = el('div', { className: 'body' });
  body.append(el('div', {}, el('span', { className: 'query-label' }, t('results.query'))));
  body.append(el('div', { className: 'muted small' }, t('results.lungQueryNote')));
  card.append(body);
  return card;
}

function lungCard(item) {
  // **外周を描いた画像を主表示にする。** 何を「肺野」として比較したのかが見えないと、
  // 脊柱・心陰影・横隔膜が正しく抜けているかを確かめられない。
  const card = hitCard(item);
  if (!item.lung_outline_url) return card;
  const figure = el('figure', { className: 'lung-figure' }, [
    lazyImage(item.lung_outline_url, t('results.lungOutlineAlt', { name: item.source_image_name || '' })),
    el('figcaption', { className: 'muted small' }, t('results.lungCaption')),
  ]);
  card.querySelector('.imgs').replaceWith(figure);
  return card;
}

function renderResults(result) {
  revokeImages();
  state.lastSearch = result;
  $('results-empty').hidden = true;
  $('results').hidden = false;

  $('result-meta').textContent = t('results.meta', {
    id: result.search_id, mode: result.mode, count: result.results.length,
  });

  const warnings = $('result-warnings');
  warnings.innerHTML = '';
  // warning_details は warnings（英語の文）を訳すための詳細で、同じ順・同じ件数。
  // 件数が合わない応答（この欄を持つ前に保存された非同期検索は既定値で埋まる）は
  // 英語の文をそのまま出す（警告を落とさないため）
  const warnings_ = result.warnings || [];
  const details = result.warning_details || [];
  const warningTexts = details.length === warnings_.length
    ? details.map((message) => serverText(message.id, message.params, message.text))
    : warnings_;
  warningTexts.forEach((warning) => warnings.append(el('li', {}, warning)));

  const models = $('result-models');
  models.innerHTML = '';
  Object.entries(result.models || {}).forEach(([key, value]) => {
    models.append(el('dt', {}, key), el('dd', {}, value));
  });
  if (result.estimate_model_version) {
    models.append(el('dt', {}, t('results.model.findingEstimate')), el('dd', {}, result.estimate_model_version));
  }
  if (result.query) {
    models.append(el('dt', {}, t('results.model.queryText')),
      el('dd', {}, result.query.text_original || t('results.imageOnly')));
    models.append(el('dt', {}, t('results.model.embeddedText')),
      el('dd', {}, result.query.text_normalized_en || '-'));
  }

  // 胸部野の類似症例（主表示）
  const main = $('chest-results');
  main.innerHTML = '';
  const block = el('div', { className: 'card region-block' });
  const head = el('div', { className: 'region-head' }, [
    el('h3', {}, t('results.chestHeading')),
    el('span', { className: 'badge' }, `${result.timing_ms} ms`),
    el('span', { className: 'badge' }, t('common.count', { count: result.results.length })),
  ]);
  const rerank = rerankBadge(result);
  if (rerank) head.append(rerank);
  block.append(head);
  block.append(el('p', { className: 'muted small' }, t('results.chestNote')));
  block.append(el('p', { className: 'muted small' }, t('results.labelNote')));
  const cards = el('div', { className: 'cards' });
  if (result.query_chest_url) cards.append(queryChestCard(result));
  if (!result.results.length) {
    block.append(cards);
    block.append(el('p', { className: 'muted small' }, t('results.noCandidates')));
  } else {
    result.results.forEach((item) => cards.append(hitCard(item)));
    block.append(cards);
  }
  main.append(block);

  // 胸郭の形が似た症例。**所見の判別ではない**ので、主表示とは別の枠に出して
  // 役割の違いを明示する。
  const lung = $('lung-results');
  lung.innerHTML = '';
  if (result.lung_shape_status && result.lung_shape_status !== 'NOT_APPLICABLE') {
    const lungBlock = el('div', { className: 'card region-block' });
    const lungHead = el('div', { className: 'region-head' }, [el('h3', {}, t('results.lungHeading'))]);
    const shapeClass = SHAPE_CLASSES[result.lung_shape_status];
    if (shapeClass) {
      lungHead.append(el('span', { className: shapeClass }, t(`results.shape.${result.lung_shape_status}`)));
    }
    if (result.lung_shape_results.length) {
      lungHead.append(el('span', { className: 'badge' },
        t('common.count', { count: result.lung_shape_results.length })));
    }
    lungBlock.append(lungHead);
    lungBlock.append(el('p', { className: 'muted small' }, t('results.lungNote')));
    if (!result.lung_shape_results.length) {
      lungBlock.append(el('p', { className: 'muted small' },
        result.lung_shape_status === 'EMPTY' ? t('results.lungEmpty') : t('results.lungFailed')));
    } else {
      const lungCards = el('div', { className: 'cards' });
      if (result.query_lung_outline_url) lungCards.append(queryLungCard(result.query_lung_outline_url));
      result.lung_shape_results.forEach((item) => lungCards.append(lungCard(item)));
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
    renderValidation(result);
    $('run-ingest').disabled = !result.valid;
  } catch (error) {
    $('ingest-error').textContent = error.message;
    $('ingest-error').hidden = false;
  }
});

function renderValidation(result) {
  state.lastValidation = result;
  $('validation-result').hidden = false;

  const counts = $('validation-counts');
  counts.innerHTML = '';
  counts.append(el('dt', {}, 'validation_id'), el('dd', {}, result.validation_id));
  counts.append(el('dt', {}, t('ingest.verdict')), el('dd', {}, result.valid ? t('ingest.pass') : t('ingest.fail')));
  Object.entries(result.counts).forEach(([key, value]) =>
    counts.append(el('dt', {}, key), el('dd', {}, String(value))));

  const issues = $('validation-issues');
  issues.innerHTML = '';
  issues.append(renderIssues(t('ingest.errors'), result.errors));
  issues.append(renderIssues(t('ingest.warnings'), result.warnings));
}

function renderIssues(title, issues) {
  const wrap = el('div', {});
  wrap.append(el('h3', {}, t('ingest.issuesHeading', { title, count: issues.length })));
  if (!issues.length) {
    wrap.append(el('p', { className: 'muted small' }, t('common.none')));
    return wrap;
  }
  const table = el('table');
  table.append(el('tr', {}, [
    el('th', {}, t('ingest.col.code')), el('th', {}, t('ingest.col.target')), el('th', {}, t('ingest.col.detail')),
  ]));
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
    $('ingest-progress-label').textContent = t('ingest.registered');
    $('ingest-progress-percent').textContent = '-';
    $('ingest-progress-detail').textContent = t('ingest.registeredDetail', { id: result.job_id });
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

const JOB_BAR_TONE = { SUCCEEDED: 'ok', PARTIAL: 'warn', FAILED: 'err', CANCELLED: 'warn' };
const JOB_FINISHED = ['SUCCEEDED', 'FAILED', 'PARTIAL', 'CANCELLED'];
/* 取消ボタンを出す状態（backend の CANCELLABLE_JOB_STATUSES と一致させる） */
const JOB_CANCELLABLE = ['PENDING', 'RUNNING'];

//: 文言は辞書の job.status.<状態> / job.field.<欄>。辞書に無い値はそのまま出す。
const jobStatusLabel = (status) => translateOr(`job.status.${status}`, status);
const progressLabel = (field) => translateOr(`job.field.${field}`, field);
const num = (value) => Number(value || 0).toLocaleString(locale());

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '-';
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return t('job.duration.hours', { hours, minutes });
  if (minutes) return t('job.duration.minutes', { minutes, seconds: total % 60 });
  return t('job.duration.seconds', { seconds: total });
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

  const parts = [
    t('job.progress.count', { done: num(done), total: num(total) }),
    t('job.progress.succeeded', { n: num(progress.succeeded) }),
    t('job.progress.failed', { n: num(progress.failed) }),
  ];
  if (progress.quarantined) parts.push(t('job.progress.quarantined', { n: num(progress.quarantined) }));
  if (progress.cancelled) parts.push(t('job.progress.cancelled', { n: num(progress.cancelled) }));
  if (!finished) {
    parts.push(t('job.progress.running', { n: num(progress.running) }), t('job.progress.remaining', { n: num(remain) }));
  }
  if (Number.isFinite(elapsed)) parts.push(t('job.progress.elapsed', { time: formatDuration(elapsed) }));
  if (!finished && rate > 0 && remain > 0) {
    parts.push(t('job.progress.eta', { time: formatDuration(remain / rate) }),
      t('job.progress.rate', { rate: rate.toFixed(1) }));
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
    track.setAttribute('aria-valuetext', t('job.progress.aria', { percent: percent.toFixed(0), done, total }));
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
      $('job-counts').append(el('dt', {}, t('common.error')), el('dd', {}, error.message));
      $('ingest-progress-detail').textContent = t('job.progress.fetchFailed', { message: error.message });
      $('ingest-progress-track').classList.remove('indeterminate');
    }
  };
  clearTimeout(state.jobTimer);
  tick();
}

function renderJob(status) {
  state.lastJobStatus = status;
  paintProgress(status);
  const counts = $('job-counts');
  counts.innerHTML = '';
  counts.append(el('dt', {}, t('jobs.jobId')), el('dd', {}, status.job_id));
  counts.append(el('dt', {}, t('jobs.status')), el('dd', {}, jobStatusLabel(status.status)));
  Object.entries(status.progress).forEach(([key, value]) =>
    counts.append(el('dt', {}, progressLabel(key)), el('dd', {}, num(value))));
  Object.entries(status.models || {}).forEach(([key, value]) =>
    counts.append(el('dt', {}, `model.${key}`), el('dd', {}, value)));

  const errors = $('job-errors');
  errors.innerHTML = '';
  if (status.error_summary?.length) {
    const table = el('table');
    table.append(el('tr', {}, [el('th', {}, t('jobs.errorCode')), el('th', {}, t('jobs.errorCount'))]));
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
    const headers = [
      t('jobs.jobId'), t('jobs.status'), progressLabel('total'), progressLabel('succeeded'),
      progressLabel('failed'), t('jobs.col.createdAt'), t('jobs.col.action'),
    ];
    table.append(el('tr', {}, headers.map((h) => el('th', {}, h))));
    (status.recent_jobs || []).forEach((job) => {
      const action = el('td');
      if (canCancel && JOB_CANCELLABLE.includes(job.status)) {
        const button = el('button', { type: 'button', className: 'danger small', textContent: t('jobs.cancel') });
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
        el('td', {}, new Date(job.created_at).toLocaleString(locale())),
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
  const message = t('jobs.cancelConfirm', { id: job.job_id.slice(0, 8), remaining: num(remaining) });
  if (!window.confirm(message)) return;
  $('recent-jobs-error').hidden = true;
  button.disabled = true;
  try {
    await api(`/api/v1/ingestions/${job.job_id}/cancel`, { method: 'POST' });
  } catch (error) {
    $('recent-jobs-error').textContent = t('jobs.cancelFailed', { message: error.message });
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
  if (!fine) return t('admin.fine.notInstalled');
  return fine >= total ? t('admin.fine.all', { fine }) : t('admin.fine.partial', { fine, total });
}

function showLoading(box) {
  box.innerHTML = '';
  box.append(el('p', { className: 'muted' }, t('common.loading')));
}

const headerRow = (keys) => el('tr', {}, keys.map((key) => el('th', {}, key ? t(key) : '')));

async function loadAdmin() {
  const box = $('admin-content');
  showLoading(box);
  try {
    const status = await api('/api/v1/admin/status');
    box.innerHTML = '';

    // ベクトル用クラスタは 1 つ（1 Coordinator + 5 Worker）。
    const cluster = status.cluster || {};
    box.append(el('h3', {}, t('admin.clusterHeading')));
    const clusterTable = el('table');
    clusterTable.append(headerRow([
      'admin.cluster.db', 'admin.cluster.coordinator', 'admin.cluster.status', 'admin.cluster.workers',
      'admin.cluster.shards', 'admin.cluster.rows', 'admin.cluster.fine', 'admin.cluster.lung',
    ]));
    const placement = cluster.shard_placement || {};
    clusterTable.append(el('tr', {}, [
      el('td', {}, cluster.database || '-'),
      el('td', {}, cluster.coordinator || '-'),
      el('td', {}, cluster.available ? t('admin.cluster.ok') : t('admin.cluster.down')),
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

    box.append(el('h3', {}, t('admin.modelsHeading')));
    const models = el('table');
    models.append(headerRow([
      'admin.model.version', 'admin.model.type', 'admin.model.name', 'admin.model.revision',
      'admin.model.dimension', 'admin.model.active', null,
    ]));
    (status.models || []).forEach((model) => {
      const button = el('button', { className: 'ghost' }, t('admin.model.activate'));
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

    box.append(el('h3', {}, t('admin.assetsHeading')));
    const chest = status.chest_assets || {};
    const estimates = status.finding_estimates || {};
    const assets = el('table');
    assets.append(headerRow([
      'admin.assets.chestCount', 'admin.assets.avgConfidence', 'admin.assets.uncertain',
      'admin.assets.estimateCount', 'admin.assets.estimateModels',
    ]));
    assets.append(el('tr', {}, [
      el('td', {}, String(chest.count ?? 0)),
      el('td', {}, String(chest.avg_confidence ?? '-')),
      el('td', {}, String(chest.uncertain ?? 0)),
      el('td', {}, String(estimates.count ?? 0)),
      el('td', {}, String(estimates.models ?? 0)),
    ]));
    box.append(assets);

    const vectors = el('table');
    vectors.append(headerRow(['admin.vectorState', 'admin.count']));
    (status.vector_states || []).forEach((row) =>
      vectors.append(el('tr', {}, [el('td', {}, row.state), el('td', {}, String(row.count))])));
    box.append(vectors);

    box.append(el('h3', {}, t('admin.queueHeading')));
    const queues = el('table');
    queues.append(headerRow(['admin.queue', 'admin.queueDepth']));
    Object.entries(status.queue_depth || {}).forEach(([queue, depth]) =>
      queues.append(el('tr', {}, [
        el('td', {}, queue), el('td', {}, depth < 0 ? t('admin.queueFailed') : String(depth)),
      ])));
    box.append(queues);

    box.append(el('p', { className: 'muted small' },
      t('admin.footer', { count: num(status.image_count), version: status.preprocess_version })));
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
    notify('forgot-notice', acceptedText(result));
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
    notify('reset-error', t('reset.invalidUrl'), true);
    return;
  }
  const password = $('reset-password').value;
  const confirmation = $('reset-password-confirm').value;
  if (password !== confirmation) {
    notify('reset-error', t('common.passwordMismatch'), true);
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
    notify('login-notice', acceptedText(result));
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
    notify('password-error', t('common.passwordMismatch'), true);
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
    notify('password-notice', acceptedText(result));
    // パスワード変更で既存トークンは失効するため、数秒後にログイン画面へ戻す
    setTimeout(() => { logout(); notify('login-notice', acceptedText(result)); }, 3000);
  } catch (error) {
    notify('password-error', error.message, true);
  } finally {
    button.disabled = false;
  }
});

/* ============================================================ ユーザー管理 */

//: 文言は辞書の roles.<役割>。辞書に無い役割はそのまま出す。
const roleLabel = (role) => translateOr(`roles.${role}`, role);

function formatDateTime(value) {
  return value ? new Date(value).toLocaleString(locale()) : '-';
}

async function loadUsers() {
  notify('users-error', '');
  const box = $('users-table');
  showLoading(box);
  try {
    const includeInactive = $('users-include-inactive').checked;
    const data = await api(`/api/v1/users?include_inactive=${includeInactive}`);
    state.users = data.users;
    box.innerHTML = '';

    const table = el('table');
    table.append(headerRow(['users.username', 'users.role', 'users.email', 'users.status', 'users.lastLogin', null]));
    data.users.forEach((user) => {
      const detail = el('button', { className: 'ghost' }, t('users.detail'));
      detail.addEventListener('click', () => showUserDetail(user.user_id));
      table.append(el('tr', {}, [
        el('td', {}, user.username),
        el('td', {}, user.role),
        el('td', {}, user.email),
        el('td', {}, user.is_active ? t('common.active') : t('common.inactive')),
        el('td', {}, formatDateTime(user.last_login_at)),
        el('td', {}, detail),
      ]));
    });
    box.append(table);
    box.append(el('p', { className: 'muted small' }, t('common.count', { count: data.total })));
  } catch (error) {
    box.innerHTML = '';
    notify('users-error', error.message, true);
  }
}

$('refresh-users').addEventListener('click', loadUsers);
$('users-include-inactive').addEventListener('change', loadUsers);

function userFormTitle() {
  const user = state.editingUserId ? state.users.find((item) => item.user_id === state.editingUserId) : null;
  const name = user ? user.username : $('user-username').value;
  $('user-form-title').textContent = state.editingUserId
    ? t('users.formTitleEdit', { name })
    : t('users.formTitleNew');
}

function openUserForm(user) {
  state.editingUserId = user ? user.user_id : null;
  $('user-form-card').hidden = false;
  $('user-detail-card').hidden = true;
  $('user-username').value = user ? user.username : '';
  userFormTitle();
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
      notify('user-form-error', t('common.passwordMismatch'), true);
      return;
    }
    if (password.length < state.policy.min_length || password.length > state.policy.max_length) {
      notify('user-form-error',
        t('users.passwordLength', { min: state.policy.min_length, max: state.policy.max_length }), true);
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
      notify('users-notice', t('users.updatedNotice'));
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
      notify('users-notice', t('users.addedNotice'));
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
    renderUserDetail(user);
    $('user-detail-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (error) {
    notify('users-error', error.message, true);
  }
}

function renderUserDetail(user) {
  state.userDetail = user;
  const box = $('user-detail');
  box.innerHTML = '';
  const byWhom = (time, who) => t('users.byWhom', { time: formatDateTime(time), who: who || '-' });
  const rows = [
    [t('users.username'), user.username],
    [t('users.role'), roleLabel(user.role)],
    [t('users.email'), user.email],
    [t('users.status'), user.is_active ? t('common.active') : t('common.inactive')],
    [t('users.memo'), user.memo || t('common.noneParen')],
    [t('users.lastLogin'), formatDateTime(user.last_login_at)],
    [t('users.passwordChanged'), formatDateTime(user.password_changed_at)],
    [t('users.created'), byWhom(user.created_at, user.created_by)],
    [t('users.updated'), byWhom(user.updated_at, user.updated_by)],
    ['user_id', user.user_id],
  ];
  rows.forEach(([key, value]) => box.append(el('dt', {}, key), el('dd', {}, value)));
}

$('user-detail-close').addEventListener('click', () => { $('user-detail-card').hidden = true; });

$('user-detail-edit').addEventListener('click', async () => {
  const user = await api(`/api/v1/users/${state.editingUserId}`);
  openUserForm(user);
});

$('user-detail-delete').addEventListener('click', async () => {
  const user = state.users.find((item) => item.user_id === state.editingUserId);
  const name = user ? user.username : state.editingUserId;
  if (!window.confirm(t('users.deleteConfirm', { name }))) return;
  notify('users-error', '');
  try {
    await api(`/api/v1/users/${state.editingUserId}`, { method: 'DELETE' });
    $('user-detail-card').hidden = true;
    notify('users-notice', t('users.deletedNotice', { name }));
    loadUsers();
  } catch (error) {
    notify('users-error', error.message, true);
  }
});

/* ------------------------------------------------------------ 言語の切り替え */

const activeTab = () => document.querySelector('.tab.active')?.dataset.tab;

/* JS が描いた文言を、手元に持っている応答から描き直す。
 * 検索の警告も描き直す。すでに出ているエラー・通知は、出した時点の言語のまま残る。 */
function refreshDynamicTexts() {
  // applySession は未ログインならログイン画面へ戻すので使わない
  // （パスワード再設定の画面で言語を切り替えても、その画面に留まる）
  renderSessionUser();
  applyPolicyHints();
  renderPreviewFacts();
  renderPhrases();
  fillFindingOptions();
  if (!$('user-form-card').hidden) userFormTitle();
  if (state.lastSearch) renderResults(state.lastSearch);
  if (state.lastValidation && !$('validation-result').hidden) renderValidation(state.lastValidation);
  if (state.lastJobStatus && !$('job-progress').hidden) renderJob(state.lastJobStatus);
  if (state.userDetail && !$('user-detail-card').hidden) renderUserDetail(state.userDetail);
  if (!state.token) return;
  const tab = activeTab();
  if (tab === 'admin') loadAdmin();
  if (tab === 'users') loadUsers();
  loadRecentJobs();
}

$('lang-select').addEventListener('change', async (event) => {
  await setLanguage(event.target.value);
  refreshDynamicTexts();
});

/* -------------------------------------------------------------- 初期化 */

async function start() {
  // 日本語は訳し漏れの受け皿を兼ねるので、どの言語でも先に読む
  try {
    i18n.fallback = await loadDictionary('ja');
  } catch (_) { /* 辞書が無ければキーがそのまま出る */ }
  await setLanguage(initialLanguage());

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
}

start();
