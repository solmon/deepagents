import React from 'react'

type TodoItem = { content: string; status?: string }

export default function AgentMessageTodo({ items }: { items: TodoItem[] }){
  return (
    <div className="todo-list">
      <strong className="todo-title">Actions</strong>
      <ul>
        {items.map((t, i) => (
          <li key={i} className={t.status === 'completed' ? 'done' : (t.status === 'in_progress' ? 'in-progress' : 'pending')}>
            <span className="todo-content">{t.content}</span>
            <span className="todo-meta">{t.status ? ` — ${t.status}` : ''}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
