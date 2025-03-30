'use client'

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import dynamic from "next/dynamic"

const WaveformPlayer = dynamic(() => import('../components/WaveformPlayer'), { ssr: false });



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
    setPlaylist(playlist.filter((song) => song.id !== id));
  };

  return (
    <div className="min-h-screen flex bg-black text-white font-sans">


      {/* Sidebar */}
      <motion.aside 
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="w-72 bg-[#121212] p-4 flex flex-col text-sm space-y-4"
      >
        <motion.h2 
          whileHover={{ scale: 1.02 }}
          className="text-lg font-bold text-white"
        >
          🎚️ Upcoming Mix Info
        </motion.h2>

        <AnimatePresence>
          {playlist.length === 0 ? (
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-gray-500"
            >
              No songs in the mix yet.
            </motion.p>
          ) : (
            <div className="space-y-4">
              <AnimatePresence>
                {playlist.map((song, i) => {
                  const next = playlist[i + 1];
                  const matchScore = Math.floor(Math.random() * 21) + 80;
                  return (
                    <motion.div
                      key={song.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -50 }}
                      transition={{ duration: 0.3 }}
                      layout
                      className="bg-[#181818] p-3 rounded-md"
                    >
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
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="text-sm bg-red-600 hover:bg-red-700 px-3 py-1 rounded mt-2"
                        onClick={() => removeFromPlaylist(song.id)}
                      >
                        ❌ Remove from Mix
                      </motion.button>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </AnimatePresence>
      </motion.aside>

      

      {/* Main */}
      <div className="flex-1 bg-gradient-to-b from-[#1f1f1f] to-black p-6">
        {/* Top Navbar */}
        <motion.div 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="flex items-center justify-between mb-8"
        >
          <div className="flex gap-2">
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="w-8 h-8 bg-black rounded-full flex items-center justify-center"
            >
              ←
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="w-8 h-8 bg-black rounded-full flex items-center justify-center"
            >
              →
            </motion.button>
          </div>
          <motion.div whileHover={{ scale: 1.01 }}>
            <input
              type="text"
              placeholder="Search for a song..."
              className="bg-[#2a2a2a] text-white px-4 py-2 rounded-full w-1/2 focus:outline-none"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </motion.div>
          <motion.img
            whileHover={{ scale: 1.1 }}
            src="https://marketplace.canva.com/EAEdeiU-IeI/1/0/1600w/canva-purple-and-red-orange-tumblr-aesthetic-chill-acoustic-classical-lo-fi-playlist-cover-jGlDSM71rNM.jpg"
            alt="User avatar"
            className="w-10 h-10 rounded-full object-cover"
          />
        </motion.div>

        {/* Playlist Section */}
        <section className="mb-10">
          <motion.h2 
            whileHover={{ scale: 1.01 }}
            className="text-2xl font-bold mb-4"
          >
            🎧 Current Transition Mix Playlist
          </motion.h2>
          {playlist.length === 0 ? (
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-gray-400"
            >
              No songs added yet.
            </motion.p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
              <AnimatePresence>
                {playlist.map((song) => (
                  <motion.div
                    key={song.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.2 }}
                    layout
                    className="bg-[#181818] rounded-lg p-4"
                  >
                    <motion.img
                      whileHover={{ scale: 1.03 }}
                      src={song.image}
                      alt={song.title}
                      className="w-full h-[160px] object-cover rounded mb-3"
                    />
                    <div className="font-semibold">{song.title}</div>
                    <div className="text-sm text-gray-400">{song.artist}</div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="text-sm bg-red-600 hover:bg-red-700 px-3 py-1 rounded mt-2"
                      onClick={() => removeFromPlaylist(song.id)}
                    >
                      ❌ Remove from Mix
                    </motion.button>
                  </motion.div>
                ))}
              </AnimatePresence>


              
            </div>

            
          )}
        </section>

        <div style={{ background: 'black', padding: '20px' }}>
      <WaveformPlayer 
        audioUrl="/song.wav"
        containerWidth="800px" // Fixed width container
        pixelsPerSecond={50}   // Adjust detail level
      />
    </div>

        {/* Total Songs Section */}
        <section className="mb-10">
          <motion.h2 
            whileHover={{ scale: 1.01 }}
            className="text-2xl font-bold mb-4"
          >
            🎵 All Songs
          </motion.h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
            {filteredSongs.map((song) => (
              <motion.div
                key={song.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ y: -5, scale: 1.02 }}
                transition={{ duration: 0.2 }}
                className="bg-[#181818] rounded-lg p-4 hover:bg-[#282828] transition duration-200"
              >
                <motion.img
                  whileHover={{ scale: 1.03 }}
                  src={song.image}
                  alt={song.title}
                  className="w-full h-[160px] object-cover rounded mb-3"
                />
                <div className="font-semibold">{song.title}</div>
                <div className="text-sm text-gray-400">{song.artist}</div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-sm bg-[#1db954] hover:bg-[#159e44] px-3 py-1 rounded"
                  onClick={() => addToPlaylist(song)}
                >
                  ➕ Add to Mix
                </motion.button>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Play Button */}
        <AnimatePresence>
          {playlist.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="text-center mt-8"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => alert("Playing your mix! 🎶")}
                className="bg-[#1db954] hover:bg-[#159e44] text-white px-8 py-3 text-lg font-bold rounded-full"
              >
                ▶️ Play Transition Mix
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        
      </div>
    </div>
  );
}