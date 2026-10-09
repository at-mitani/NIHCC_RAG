# postmaster が PGPASSWORD を引き継がないようにする。
#
# **このファイルは実行せず source される。** docker-entrypoint.sh の
# `docker_process_init_files` は、実行権の無い *.sh を `.` で読み込む。だから
# ここでの `unset` が entrypoint 自身の環境に効き、最後の `exec postgres` へ
# 引き継がれない。実行権を付けると別プロセスになり、何も起きなくなる。
#
# ## なぜ要るのか
#
# docker-entrypoint.sh は**データディレクトリが空のときだけ** PGPASSWORD を
# export する（初期化中の psql が local 接続で要るため）。そのまま postgres を
# exec するので、**初回起動の postmaster だけが PGPASSWORD を持つ**。
#
# libpq は PGPASSWORD を .pgpass より優先する。Citus Community は
# Coordinator → Worker の内部接続を「クエリを実行中のロール」で張り、その
# パスワードを .pgpass から取る設計なので、PGPASSWORD が
# 残っていると **app_ingest でも app_search でも postgres のパスワードを
# 送ってしまい**、Worker が撥ねる。
#
#   FATAL: password authentication failed for user "app_ingest"
#
# 初期化直後にだけ起きて、コンテナを再起動すると直るので、気づきにくい。
# 取り込みは VECTOR_STORE_PARTIAL_FAILURE で失敗する。
#
# 初期化はこの時点で終わっており、このあと entrypoint がするのは一時サーバの
# 停止と postgres の exec だけなので、消して困るものは無い。
unset PGPASSWORD
