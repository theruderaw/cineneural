document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const movieTitle = urlParams.get('title');
    
    if (!movieTitle) {
        window.location.href = '/';
        return;
    }

    const loader = document.getElementById('detail-loader');
    const content = document.getElementById('movie-content');

    try {
        const response = await fetch(`/api/movie/${encodeURIComponent(movieTitle)}`);
        if (!response.ok) throw new Error('Movie not found');
        
        const movie = await response.json();
        
        // Populate Data
        document.getElementById('movie-title').innerText = movie.Title;
        document.getElementById('movie-meta-header').innerText = `${movie['Release Year']} • ${movie['Origin/Ethnicity']}`;
        document.getElementById('movie-plot').innerText = movie.Plot;
        document.getElementById('movie-director').innerText = movie.Director || 'N/A';
        document.getElementById('movie-cast').innerText = movie.Cast || 'N/A';
        document.getElementById('movie-origin').innerText = movie['Origin/Ethnicity'];

        // Genre Pills (Tailwind-Compliant)
        const genreContainer = document.getElementById('genre-container');
        const genres = movie.Genre.split(',').map(g => g.trim()).filter(g => g);
        genreContainer.innerHTML = genres.map(g => `
            <span class="bg-white/5 border border-white/10 px-4 py-1.5 rounded-full text-[10px] text-zinc-400 font-black uppercase tracking-widest hover:text-white hover:border-white/30 transition-all cursor-default">
                ${g}
            </span>
        `).join('');

        // Show content with Tailwind classes
        loader.classList.add('hidden');
        content.classList.remove('hidden');
        
        // Update Page Title
        document.title = `${movie.Title} | CineNeural`;

    } catch (err) {
        console.error('Error fetching movie:', err);
        document.body.innerHTML = `
            <div class="container mx-auto px-6 py-40 text-center">
                <h1 class="text-4xl font-black text-white mb-4">404: Neural Path Expired</h1>
                <p class="text-zinc-500 mb-12">The film index you are requesting is no longer reachable in the latent space.</p>
                <a href="/" class="inline-flex items-center gap-2 text-white bg-zinc-900 border border-white/10 px-8 py-3 rounded-full hover:bg-white hover:text-black transition-all font-black text-xs uppercase tracking-widest">
                    Go Back Home
                </a>
            </div>
        `;
    }
});
