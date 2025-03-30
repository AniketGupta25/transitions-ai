export default function SongList({ songs, onAdd }: { songs: any[], onAdd: (song: any) => void }) {
    if (!songs.length) return null
  
    return (
      <div>
        <h2 className="text-xl font-semibold mt-4 mb-2">Search Results</h2>
        <ul className="space-y-2">
          {songs.map((song) => (
            <li key={song.id} className="flex justify-between items-center border p-2 rounded">
              <div>{song.title} — {song.artist}</div>
              <button onClick={() => onAdd(song)} className="text-sm bg-green-500 text-white px-3 py-1 rounded">
                Add
              </button>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  