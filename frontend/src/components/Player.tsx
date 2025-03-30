import { useState, useRef } from "react"

export default function Player({ playlist }: { playlist: any[] }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [current, setCurrent] = useState(0)

  const handlePlay = () => {
    const next = playlist[current]
    if (next && audioRef.current) {
      audioRef.current.src = `https://example.com/audio/${next.id}.mp3` // replace with real URL
      audioRef.current.play()
    }
  }

  return (
    <div className="mt-6">
      <button
        className="bg-purple-600 text-white px-4 py-2 rounded"
        onClick={handlePlay}
        disabled={!playlist.length}
      >
        ▶️ Play Transition Mix
      </button>
      <audio ref={audioRef} onEnded={() => setCurrent((i) => (i + 1) % playlist.length)} />
    </div>
  )
}
