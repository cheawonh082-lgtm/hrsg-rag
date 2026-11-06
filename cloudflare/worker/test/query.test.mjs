// node --test cloudflare/worker/test — Python(rag/lang.py · rag/glossary.py)과 같은 결과인지
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import { detectLang, englishTerms, expandQuery, glossaryEntries, sameSecret, truncateNormalize, weightedRrf } from '../src/query.js'

const glossary = JSON.parse(readFileSync(new URL('../src/glossary.json', import.meta.url), 'utf-8'))
const entries = glossaryEntries(glossary)

test('언어 판별: 영어 약어가 섞인 한국어 질문도 ko', () => {
  assert.equal(detectLang('HRSG 핀치 포인트란?'), 'ko')
  assert.equal(detectLang('What is a pinch point?'), 'en')
  assert.equal(detectLang('123 ?!'), 'en')
})

test('용어 확장: 띄어쓰기 무시, 긴 용어 우선 (Python 테스트와 같은 기대값)', () => {
  assert.deepEqual(englishTerms('고압증발기 튜브의 침착물', entries), ['HP evaporator', 'tube', 'deposit deposition'])
  assert.ok(expandQuery('배열 회수 보일러 핀치 포인트', entries).includes('pinch point'))
  assert.equal(expandQuery('What is a pinch point?', entries), 'What is a pinch point?')
})

test('비밀키 비교', () => {
  assert.ok(sameSecret('Bearer abc', 'Bearer abc'))
  assert.ok(!sameSecret('Bearer abd', 'Bearer abc'))
  assert.ok(!sameSecret('Bearer ab', 'Bearer abc'))
})

test('가중 RRF: Python weighted_rrf와 같은 순서·점수 (tests/test_rag_api.py와 같은 예)', () => {
  const fused = weightedRrf({ bge: ['a', 'b', 'c'], qwen: ['b', 'a', 'd'] }, 60, { bge: 1, qwen: 1 })
  assert.deepEqual(fused.map(([id]) => id), ['a', 'b', 'c', 'd']) // a·b 동점 → 최고 순위 1위끼리 → ID 순
  assert.ok(Math.abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-12)
  assert.deepEqual(weightedRrf({ bge: ['a'], bm25: ['z'] }, 60, { bge: 1 }).map(([id, s]) => [id, s]), [['a', 1 / 61], ['z', 0]])
})

test('MRL: 앞 차원만 남기고 정규화', () => {
  const v = truncateNormalize([3, 4, 100], 2)
  assert.deepEqual(v, [0.6, 0.8])
})
