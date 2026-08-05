import React, { useEffect, useState } from 'react'
import { Trash2, FolderOpen, X, Plus } from 'lucide-react'
import { listSessions, loadSession, deleteSession, listProjects, loadProject, deleteProject } from '../api/client'
import { useEditorStore } from '../store/editorStore'
import { useBlobUrl } from '../hooks/useBlobUrl'
import { baseUrl } from '../config'
import { LayerData } from '../types'
import { toast } from 'react-toastify'

interface Props { onClose: () => void; onNew: () => void }

type Tab = 'sessions' | 'projects'

interface SessionInfo { session_id: string; image_path: string; layer_count: number; mtime: number }

const SessionThumb: React.FC<{ image_path: string }> = ({ image_path }) => {
  const url = useBlobUrl(`${baseUrl}/temp/${image_path.replace(/^\/temp\//, '').replace(/^\//, '')}`)
  return url ? <img src={url} className="w-12 h-12 object-cover rounded flex-shrink-0" /> : <div className="w-12 h-12 bg-dark-600 rounded flex-shrink-0" />
}

export const ProjectManager: React.FC<Props> = ({ onClose, onNew }) => {
  const [tab, setTab] = useState<Tab>('sessions')
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [projects, setProjects] = useState<string[]>([])
  const { setSession, setLayers, reset } = useEditorStore()

  useEffect(() => {
    listSessions().then(r => setSessions(r.sessions)).catch(() => {})
    listProjects().then(r => setProjects(r.projects)).catch(() => {})
  }, [])

  const openSession = async (s: SessionInfo) => {
    const { session } = await loadSession(s.session_id)
    const imageUrl = `${baseUrl}/temp/${session.image_path.replace(/^\//, '')}`
    const blob = await fetch(imageUrl, { headers: { 'ngrok-skip-browser-warning': '1' } }).then(r => r.blob())
    const blobUrl = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => {
      setSession(session.session_id, session.image_path, blobUrl, img.naturalWidth, img.naturalHeight)
      setLayers(session.layers as LayerData[])
      toast.info('Session loaded')
      onClose()
    }
    img.onerror = () => {
      setSession(session.session_id, session.image_path, imageUrl, 1920, 1080)
      setLayers(session.layers as LayerData[])
      onClose()
    }
    img.src = blobUrl
  }

  const openProject = async (name: string) => {
    const data: any = await loadProject(name)
    setSession(data.session_id, data.original_image_path, '', data.canvas_width, data.canvas_height)
    setLayers(data.layers)
    toast.info(`Project "${name}" loaded`)
    onClose()
  }

  const removeSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await deleteSession(id)
    setSessions(s => s.filter(x => x.session_id !== id))
  }

  const removeProject = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await deleteProject(name)
    setProjects(p => p.filter(x => x !== name))
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-dark-800 rounded-xl w-[480px] max-h-[70vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-dark-600">
          <div className="flex gap-2">
            {(['sessions', 'projects'] as Tab[]).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-3 py-1 rounded text-sm capitalize ${tab === t ? 'bg-accent text-white' : 'text-gray-400 hover:text-white'}`}>
                {t}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={onNew} className="flex items-center gap-1 px-3 py-1 rounded bg-dark-600 hover:bg-dark-500 text-sm text-gray-300">
              <Plus size={14} /> New
            </button>
            <button onClick={onClose} className="text-gray-400 hover:text-white"><X size={18} /></button>
          </div>
        </div>

        {/* List */}
        <div className="overflow-y-auto flex-1 p-2 space-y-1">
          {tab === 'sessions' && (sessions.length === 0
            ? <p className="text-gray-500 text-sm text-center py-8">No sessions found</p>
            : sessions.map(s => (
              <div key={s.session_id} onClick={() => openSession(s)}
                className="flex items-center gap-3 p-2 rounded hover:bg-dark-600 cursor-pointer group">
                <SessionThumb image_path={s.image_path} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{s.image_path.split('/').pop()}</p>
                  <p className="text-xs text-gray-500">{s.layer_count} layers · {new Date(s.mtime * 1000).toLocaleString()}</p>
                </div>
                <button onClick={e => removeSession(s.session_id, e)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 p-1">
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
          {tab === 'projects' && (projects.length === 0
            ? <p className="text-gray-500 text-sm text-center py-8">No saved projects</p>
            : projects.map(name => (
              <div key={name} onClick={() => openProject(name)}
                className="flex items-center gap-3 p-3 rounded hover:bg-dark-600 cursor-pointer group">
                <FolderOpen size={20} className="text-accent flex-shrink-0" />
                <span className="flex-1 text-sm text-white">{name}</span>
                <button onClick={e => removeProject(name, e)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 p-1">
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
