'use client'

import { useState } from "react"

export default function SearchBar({ setResults }: { setResults: (res: any[]) => void }) {
  const [query, setQuery] = useState("")

  const handleSearch = async () => {
    // Mock response; integrate real search later
    const mockSongs = [
      { id: 1, title: "Song A", artist: "Artist 1" },
      { id: 2, title: "Song B", artist: "Artist 2" },
    ]
    setResults(mockSongs.filter(song => song.title.toLowerCase().includes(query.toLowerCase())))
  }

  return (
    <div className="flex gap-2">
      <input
        type="text"
        className="flex-1 border rounded px-3 py-2"
        placeholder="Search for songs..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSearch} className="bg-blue-600 text-white px-4 py-2 rounded">
        Search
      </button>
    </div>
  )
}
