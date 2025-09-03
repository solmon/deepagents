import React from 'react'

type SearchResult = { url?: string; title?: string; content?: string; score?: number }

export default function AgentMessageSearch({ payload }: { payload: any }){
  // payload may be the parsed payload object which can contain query, results, images, etc.
  const query = payload?.query || payload?.q || null
  const results: SearchResult[] = payload?.results || []

  return (
    <div className="search-data">
      {query && <div className="search-query"><strong>Search:</strong> {query}</div>}
      <div className="search-results">
        {results.length === 0 ? (
          <div className="muted">No results found.</div>
        ) : (
          results.map((r, i) => (
            <div key={i} className="search-result-card">
              <a href={r.url} target="_blank" rel="noreferrer" className="search-result-title">{r.title || r.url}</a>
              <p className="search-result-snippet">{r.content || ''}</p>
              <div className="search-result-meta"><span className="score">{typeof r.score === 'number' ? r.score.toFixed(2) : ''}</span></div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
