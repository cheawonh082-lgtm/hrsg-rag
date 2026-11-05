// HRSG 논문 검색 Worker — 배포판 앱(Render)이 쓰는 검색 서버.
//
//   POST /search  {"query", "top_n"}  → RAG API(api/app.py)와 같은 형식의 상위 문단
//   GET  /health                      → 코퍼스 규모·검색 방식
//
// 질문 임베딩: Workers AI @cf/baai/bge-m3 (코퍼스 벡터와 같은 모델) · 검색: Vectorize(cosine)
// 문단 본문·서지 정보는 Vectorize 메타데이터에 있다 (scripts/cloudflare_sync.py upload).
// 모든 요청은 Authorization: Bearer <RAG_API_KEY> 가 필요하다 (wrangler secret put RAG_API_KEY).

import GLOSSARY from './glossary.json'
import STATS from './stats.json'
import { detectLang, expandQuery, glossaryEntries, sameSecret } from './query.js'

const MODEL = '@cf/baai/bge-m3'
const ENTRIES = glossaryEntries(GLOSSARY)
const MAX_TOP_N = 20 // Vectorize: 메타데이터를 함께 받으면 topK ≤ 50
const METHOD_LABEL = 'bge-m3 (Workers AI + Vectorize)'

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } })

function vectorFrom(result) {
  for (const key of ['response', 'data']) {
    const v = result?.[key]
    if (Array.isArray(v) && Array.isArray(v[0])) return v[0]
  }
  throw new Error(`Workers AI 응답에서 임베딩을 찾지 못했습니다: ${Object.keys(result ?? {})}`)
}

function toResult(match, rank) {
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
    score: Math.round(match.score * 1e6) / 1e6,
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
  const embedded = await env.AI.run(MODEL, { contexts: [{ text: searchQuery }] })
  const found = await env.VECTORIZE.query(vectorFrom(embedded), { topK: topN, returnMetadata: 'all' })
  return json({
    query,
    query_lang: lang,
    search_query: searchQuery,
    method: 'cf',
    method_label: METHOD_LABEL,
    top_n: topN,
    results: (found.matches ?? []).map((m, i) => toResult(m, i + 1)),
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
    methods: { cf: true },
    missing: {},
    models: { bge: MODEL },
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
