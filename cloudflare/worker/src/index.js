// HRSG 논문 검색 Worker — 배포판 앱(Render)이 쓰는 검색 서버.
//
//   POST /search  {"query", "top_n"}  → RAG API(api/app.py)와 같은 형식의 상위 문단
//   GET  /health                      → 코퍼스 규모·검색 방식
//
// 검색: 로컬 RAG API 4안과 같은 bge:qwen:BM25 = 1:1:0 — 두 검색기 상위 top_k(50)개를 가중 RRF(k=60)로 합친다.
//   bge  — Workers AI @cf/baai/bge-m3 → Vectorize hrsg-bge-m3 (1,024차원)
//   qwen — Workers AI @cf/qwen/qwen3-embedding-0.6b (질문 instruction 포함) → 앞 512차원 → Vectorize hrsg-qwen3-06b-512
// 문단 본문·서지 정보는 두 인덱스 모두 Vectorize 메타데이터에 있다 (scripts/cloudflare_sync.py upload).
// 가중치·instruction 등은 search_config.json (scripts/cloudflare_sync.py worker-data가 rag_config에서 만든다).
// 모든 요청은 Authorization: Bearer <RAG_API_KEY> 가 필요하다 (wrangler secret put RAG_API_KEY).

import GLOSSARY from './glossary.json'
import CONFIG from './search_config.json'
import STATS from './stats.json'
import { detectLang, expandQuery, glossaryEntries, sameSecret, truncateNormalize, weightedRrf } from './query.js'

const MODELS = { bge: '@cf/baai/bge-m3', qwen: '@cf/qwen/qwen3-embedding-0.6b' }
const ENTRIES = glossaryEntries(GLOSSARY)
const MAX_TOP_N = 20
const TOP_K = Math.min(CONFIG.top_k, 50) // Vectorize: 메타데이터를 함께 받으면 topK ≤ 50
const METHOD_LABEL = 'bge-m3 + Qwen3 → 가중 RRF (4안, bge:qwen:BM25 = 1:1:0)'

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } })

function vectorFrom(result) {
  for (const key of ['response', 'data']) {
    const v = result?.[key]
    if (Array.isArray(v) && Array.isArray(v[0])) return v[0]
  }
  throw new Error(`Workers AI 응답에서 임베딩을 찾지 못했습니다: ${Object.keys(result ?? {})}`)
}

async function retrieve(env, searchQuery) {
  const [bgeOut, qwenOut] = await Promise.all([
    env.AI.run(MODELS.bge, { contexts: [{ text: searchQuery }] }),
    env.AI.run(MODELS.qwen, { queries: [searchQuery], instruction: CONFIG.qwen_instruction }),
  ])
  const opts = { topK: TOP_K, returnMetadata: 'all' }
  const [bge, qwen] = await Promise.all([
    env.VECTORIZE.query(vectorFrom(bgeOut), opts),
    env.VECTORIZE_QWEN.query(truncateNormalize(vectorFrom(qwenOut), CONFIG.qwen_dims), opts),
  ])
  return { bge: bge.matches ?? [], qwen: qwen.matches ?? [] }
}

function toResult(match, rank, score, retrievers) {
  const m = match.metadata ?? {}
  return {
    rank,
    id: match.id,
    doc_id: m.doc_id,
    title: m.title ?? '',
    authors: m.authors ? String(m.authors).split('; ').filter(Boolean) : [],
    year: m.year ?? null,
    venue: m.venue ?? '',
    doi: m.doi ?? '',
    url: m.url ?? '',
    section: m.section ?? '',
    page_start: m.page_start ?? null,
    page_end: m.page_end ?? null,
    content_type: m.content_type ?? 'text',
    source_type: m.source_type ?? '',
    lang: m.lang ?? '',
    doc_lang: m.doc_lang ?? '',
    text: m.text ?? '',
    score: Math.round(score * 1e6) / 1e6,
    retrievers,
  }
}

async function search(request, env) {
  let body
  try {
    body = await request.json()
  } catch {
    return json({ detail: 'JSON 본문이 필요합니다' }, 400)
  }
  const query = typeof body?.query === 'string' ? body.query.trim() : ''
  if (!query || query.length > 2000) return json({ detail: 'query는 1~2000자여야 합니다' }, 422)
  const topN = Math.min(Math.max(parseInt(body.top_n ?? 5, 10) || 5, 1), MAX_TOP_N)

  const lang = detectLang(query)
  const searchQuery = lang === 'ko' ? expandQuery(query, ENTRIES) : query
  const weights = CONFIG.weights[lang]
  const hits = await retrieve(env, searchQuery)
  const byId = new Map()
  const ranks = {}
  for (const [name, matches] of Object.entries(hits)) {
    ranks[name] = new Map(matches.map((m, i) => [m.id, { rank: i + 1, score: Math.round(m.score * 1e4) / 1e4 }]))
    for (const m of matches) if (!byId.has(m.id)) byId.set(m.id, m)
  }
  const fused = weightedRrf(Object.fromEntries(Object.entries(hits).map(([n, ms]) => [n, ms.map((m) => m.id)])), CONFIG.rrf_k, weights)
  const results = fused.slice(0, topN).map(([id, score], i) => {
    const retrievers = Object.fromEntries(Object.keys(hits).filter((n) => ranks[n].has(id)).map((n) => [n, ranks[n].get(id)]))
    return toResult(byId.get(id), i + 1, score, retrievers)
  })
  return json({
    query,
    query_lang: lang,
    search_query: searchQuery,
    method: '4',
    method_label: METHOD_LABEL,
    weights,
    top_k: TOP_K,
    top_n: topN,
    results,
  })
}

async function health(env) {
  let vectors = null
  try {
    const info = await env.VECTORIZE.describe()
    vectors = info.vectorCount ?? info.vectorsCount ?? null
  } catch {
    // 인덱스 정보를 못 읽어도 상태는 돌려준다
  }
  return json({
    status: 'ok',
    corpus: { documents: STATS.documents, paragraphs: STATS.paragraphs, version: STATS.corpus_version, vectors },
    methods: { 4: true },
    missing: {},
    models: { ...MODELS, qwen_dims: CONFIG.qwen_dims },
    weights: CONFIG.weights,
  })
}

export default {
  async fetch(request, env) {
    const auth = request.headers.get('Authorization') ?? ''
    if (!env.RAG_API_KEY || !sameSecret(auth, `Bearer ${env.RAG_API_KEY}`)) return json({ detail: 'unauthorized' }, 401)
    const { pathname } = new URL(request.url)
    try {
      if (request.method === 'POST' && pathname === '/search') return await search(request, env)
      if (request.method === 'GET' && pathname === '/health') return await health(env)
      return json({ detail: 'not found' }, 404)
    } catch (e) {
      console.error(e)
      return json({ detail: '검색 중 오류가 났습니다' }, 502)
    }
  },
}
