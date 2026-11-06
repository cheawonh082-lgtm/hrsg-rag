// 질의 전처리 — rag/lang.py · rag/glossary.py와 같은 규칙 (Python 쪽을 바꾸면 여기도 맞춘다)

const HANGUL = /[가-힣ㄱ-ㅎㅏ-ㅣ]/g
const LATIN = /[A-Za-z]/g
export const KO_RATIO = 0.3

/** 'ko' | 'en' — 한글 음절 / (한글 + 라틴 문자) ≥ 0.3 이면 ko */
export function detectLang(text, threshold = KO_RATIO) {
  const h = (text.match(HANGUL) || []).length
  const total = h + (text.match(LATIN) || []).length
  return total && h / total >= threshold ? 'ko' : 'en'
}

const squash = (s) => s.replace(/\s+/g, '')

/** 사전 → (띄어쓰기 없앤 키, 영어) 목록, 긴 키부터 */
export function glossaryEntries(glossary) {
  return Object.entries(glossary)
    .map(([k, v]) => [squash(k), v])
    .sort((a, b) => b[0].length - a[0].length)
}

/** 질문에 나온 한국어 용어의 영어 표현 — 긴 용어가 먼저 잡히고, 잡힌 부분은 짧은 용어가 다시 쓰지 않는다 */
export function englishTerms(query, entries) {
  const text = squash(query)
  const taken = new Array(text.length).fill(false)
  const found = []
  for (const [key, en] of entries) {
    let start = text.indexOf(key)
    while (start !== -1) {
      const end = start + key.length
      if (!taken.slice(start, end).some(Boolean)) {
        taken.fill(true, start, end)
        found.push([start, en])
      }
      start = text.indexOf(key, end)
    }
  }
  found.sort((a, b) => a[0] - b[0])
  return [...new Set(found.map(([, en]) => en))]
}

export function expandQuery(query, entries) {
  const terms = englishTerms(query, entries)
  return terms.length ? `${query} ${terms.join(' ')}` : query
}

/** 앱(hrsg-reference-app)과 같은 비밀키인지 — 길이와 무관하게 끝까지 비교 */
export function sameSecret(a, b) {
  const x = new TextEncoder().encode(a)
  const y = new TextEncoder().encode(b)
  let diff = x.length ^ y.length
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0)
  return diff === 0
}

/** rag/rag_engine.py weighted_rrf와 같다 — rankings: {검색기: [문단 ID, 1위부터]}, 가중치 없는 검색기는 0.
 *  동점이면 어느 검색기에서든 더 높은 순위였던 문단, 그다음 ID 순. 반환: [[ID, 점수], …] */
export function weightedRrf(rankings, k, weights) {
  const score = new Map()
  const best = new Map()
  for (const [name, ranking] of Object.entries(rankings)) {
    const w = weights?.[name] ?? 0
    ranking.forEach((id, i) => {
      const rank = i + 1
      score.set(id, (score.get(id) ?? 0) + w / (k + rank))
      best.set(id, Math.min(rank, best.get(id) ?? rank))
    })
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1] || best.get(a[0]) - best.get(b[0]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
}

/** Qwen3-Embedding MRL — 앞 dims 차원만 남기고 L2 정규화 (scripts/cloudflare_sync.py mrl과 같다) */
export function truncateNormalize(vector, dims) {
  const v = vector.slice(0, dims)
  const norm = Math.hypot(...v)
  return v.map((x) => x / norm)
}
