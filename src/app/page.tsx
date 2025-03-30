'use client'

import { useState } from "react"

type Song = {
  id: number
  title: string
  artist: string
  image: string
}

const ALL_SONGS: Song[] = [
  { id: 1, title: "Blinding Lights", artist: "The Weeknd", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 2, title: "Levitating", artist: "Dua Lipa", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 3, title: "Save Your Tears", artist: "The Weeknd", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 4, title: "Heat Waves", artist: "Glass Animals", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 5, title: "Stay", artist: "The Kid LAROI, Justin Bieber", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 6, title: "Peaches", artist: "Justin Bieber", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 7, title: "Dance Monkey", artist: "Tones And I", image: "https://marketplace.canva.com/EAEdft48JIs/1/0/1600w/canva-orange-skyline-tumblr-aesthetic-love-songs-playlist-cover-mCNRRGaWFgU.jpg" },
  { id: 8, title: "Industry Baby", artist: "Lil Nas X, Jack Harlow", image: "/song8.jpg" },
]

export default function Home() {
  const [search, setSearch] = useState("");
  const [playlist, setPlaylist] = useState<Song[]>([]);

  const filteredSongs = ALL_SONGS.filter(
    (song) =>
      song.title.toLowerCase().includes(search.toLowerCase()) ||
      song.artist.toLowerCase().includes(search.toLowerCase())
  );

  const addToPlaylist = (song: Song) => {
    if (!playlist.find((s) => s.id === song.id)) {
      setPlaylist([...playlist, song]);
    }
  };

  const removeFromPlaylist = (id: number) => {
    setPlaylist(playlist.filter((song) => song.id !== id)); // Removes the song with the given ID
  };

  return (
    <div className="min-h-screen flex bg-black text-white font-sans">
      {/* Sidebar */}
      <aside className="w-72 bg-[#121212] p-4 flex flex-col text-sm space-y-4">
        <h2 className="text-lg font-bold text-white">🎚️ Upcoming Mix Info</h2>

        {playlist.length === 0 ? (
          <p className="text-gray-500">No songs in the mix yet.</p>
        ) : (
          <div className="space-y-4">
            {playlist.map((song, i) => {
              const next = playlist[i + 1];
              const matchScore = Math.floor(Math.random() * 21) + 80; // 80–100
              return (
                <div key={song.id} className="bg-[#181818] p-3 rounded-md">
                  <div className="font-semibold text-white">{song.title}</div>
                  <div className="text-gray-400 text-xs mb-2">{song.artist}</div>

                  {next && (
                    <>
                      <div className="text-green-400 font-medium mb-1">
                        ➡ Transition to: {next.title}
                      </div>
                      <div className="text-gray-300">
                        Match Score:{" "}
                        <span className="font-semibold text-white">
                          {matchScore}%
                        </span>
                      </div>
                      <div className="text-gray-400 text-xs italic">
                        Transition: Smooth tempo blend
                      </div>
                    </>
                  )}
                  {/* Remove Button */}
                  <button
                    className="text-sm bg-red-600 hover:bg-red-700 px-3 py-1 rounded mt-2"
                    onClick={() => removeFromPlaylist(song.id)}
                  >
                    ❌ Remove from Mix
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </aside>

      {/* Main */}
      <div className="flex-1 bg-gradient-to-b from-[#1f1f1f] to-black p-6">
        {/* Top Navbar */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex gap-2">
            <button className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
              ←
            </button>
            <button className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
              →
            </button>
          </div>
          <input
            type="text"
            placeholder="Search for a song..."
            className="bg-[#2a2a2a] text-white px-4 py-2 rounded-full w-1/2 focus:outline-none"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <img
            src="https://marketplace.canva.com/EAEdeiU-IeI/1/0/1600w/canva-purple-and-red-orange-tumblr-aesthetic-chill-acoustic-classical-lo-fi-playlist-cover-jGlDSM71rNM.jpg"
            alt="User avatar"
            className="w-10 h-10 rounded-full object-cover"
          />
        </div>

        {/* Playlist Section */}
        <section className="mb-10">
          <h2 className="text-2xl font-bold mb-4">🎧 Current Transition Mix Playlist</h2>
          {playlist.length === 0 ? (
            <p className="text-gray-400">No songs added yet.</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
              {playlist.map((song) => (
                <div key={song.id} className="bg-[#181818] rounded-lg p-4">
                  <img
                    src={song.image}
                    alt={song.title}
                    className="w-full h-[160px] object-cover rounded mb-3"
                  />
                  <div className="font-semibold">{song.title}</div>
                  <div className="text-sm text-gray-400">{song.artist}</div>
                  {/* Remove Button */}
                  <button
                    className="text-sm bg-red-600 hover:bg-red-700 px-3 py-1 rounded mt-2"
                    onClick={() => removeFromPlaylist(song.id)}
                  >
                    ❌ Remove from Mix
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Total Songs Section */}
        <section className="mb-10">
          <h2 className="text-2xl font-bold mb-4">🎵 All Songs</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
            {filteredSongs.map((song) => (
              <div
                key={song.id}
                className="bg-[#181818] rounded-lg p-4 hover:bg-[#282828] transition duration-200"
              >
                <img
                  src={song.image}
                  alt={song.title}
                  className="w-full h-[160px] object-cover rounded mb-3"
                />
                <div className="font-semibold">{song.title}</div>
                <div className="text-sm text-gray-400">{song.artist}</div>
                {/* Add Button */}
                <button
                  className="text-sm bg-[#1db954] hover:bg-[#159e44] px-3 py-1 rounded"
                  onClick={() => addToPlaylist(song)}
                >
                  ➕ Add to Mix
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Play Button */}
        {playlist.length > 0 && (
          <div className="text-center mt-8">
            <button
              onClick={() => alert("Playing your mix! 🎶")}
              className="bg-[#1db954] hover:bg-[#159e44] text-white px-8 py-3 text-lg font-bold rounded-full"
            >
              ▶️ Play Transition Mix
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

