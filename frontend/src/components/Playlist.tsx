export default function Playlist({ songs }: { songs: any[] }) {
    if (!songs.length) return null
  
    return (
      <div>
        <h2 className="text-xl font-semibold mt-6 mb-2">Your Playlist</h2>
        <ul className="space-y-1">
          {songs.map(song => (
            <li key={song.id} className="border p-2 rounded">
              {song.title} — {song.artist}
            </li>
          ))}
        </ul>
      </div>
    )
  }
  