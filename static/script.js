const movieInput = document.getElementById('movie-input');
const autocompleteResults = document.getElementById('autocomplete-results');
const resultsContainer = document.getElementById('results-container');
const loader = document.getElementById('loader');
const controlsPanel = document.getElementById('controls-panel');
const searchBtn = document.getElementById('search-btn');
const logicStatus = document.getElementById('logic-status');

let typingTimer;
let debounceTimer;
let currentMovie = "";
const doneTypingInterval = 300;

const sliders = {
    plot: document.getElementById('w-plot'),
    genre: document.getElementById('w-genre'),
    directorToggle: document.getElementById('w-dir-toggle'),
    origin: document.getElementById('w-origin'),
    year: document.getElementById('w-year'),
    concept: document.getElementById('w-concept')
};

// 1. Live Autocomplete (Glossy Monochrome)
movieInput.addEventListener('input', () => {
    clearTimeout(typingTimer);
    const query = movieInput.value.trim();
    if (query.length < 2) {
        autocompleteResults.classList.add('hidden');
        return;
    }
    typingTimer = setTimeout(async () => {
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, limit: 12 })
            });
            const data = await response.json();
            if (data.results.length > 0) {
                autocompleteResults.innerHTML = data.results
                    .map(title => `<div class="p-5 hover:bg-white/[0.05] cursor-pointer transition-all border-b border-white/5 last:border-0 text-sm font-medium tracking-tight text-zinc-300 hover:text-white hover:pl-8 active:bg-white/[0.08]">${title}</div>`)
                    .join('');
                autocompleteResults.classList.remove('hidden');
            } else {
                autocompleteResults.classList.add('hidden');
            }
        } catch (err) { console.error('Search error:', err); }
    }, doneTypingInterval);
});

// 2. Click Interaction
autocompleteResults.addEventListener('click', (e) => {
    const item = e.target.closest('div');
    if (item) {
        const title = item.innerText;
        movieInput.value = title;
        currentMovie = title;
        autocompleteResults.classList.add('hidden');
        controlsPanel.classList.remove('hidden');
        getRecommendations(title);
    }
});

searchBtn.addEventListener('click', () => {
    const title = movieInput.value.trim();
    if (title) {
        currentMovie = title;
        autocompleteResults.classList.add('hidden');
        controlsPanel.classList.remove('hidden');
        getRecommendations(title);
    }
});

// 3. Slider logic
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', () => {
        const labelSpan = document.getElementById(`val-${key}`);
        if (labelSpan) labelSpan.innerText = sliders[key].value;
        
        logicStatus.innerText = "Recalibrating Neural Net...";
        logicStatus.parentElement.querySelector('div').classList.replace('bg-white', 'bg-zinc-500');
        
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (currentMovie) getRecommendations(currentMovie);
        }, 600);
    });
});

function getWeights() {
    return {
        plot: sliders.plot.value / 100,
        genre: sliders.genre.value / 100,
        director: sliders.directorToggle.checked ? 0.30 : 0.0,
        origin: sliders.origin.value / 100,
        year: sliders.year.value / 100,
        concept: sliders.concept.value / 100
    };
}

// 4. Recommendation Engine Hook
async function getRecommendations(title) {
    loader.classList.remove('hidden');
    resultsContainer.classList.add('opacity-0');
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, top_n: 12, weights: getWeights() })
        });
        const data = await response.json();
        renderResults(data.recommendations);
        logicStatus.innerText = "Calibration Synced";
        logicStatus.parentElement.querySelector('div').classList.replace('bg-zinc-500', 'bg-white');
    } catch (err) {
        console.error('API Error:', err);
        resultsContainer.innerHTML = `<div class="col-span-full py-24 text-center"><p class="text-white font-black uppercase tracking-widest text-xs">Neural Disconnect: Dataset unreachable.</p></div>`;
        resultsContainer.classList.remove('opacity-0');
    } finally {
        loader.classList.add('hidden');
    }
}

function renderResults(movies) {
    resultsContainer.innerHTML = movies.map((movie, index) => `
        <div class="animate-reveal bg-zinc-950/40 border border-white/5 rounded-[2.5rem] p-8 hover:bg-zinc-950 hover:border-white/20 transition-all duration-700 cursor-pointer group shadow-2xl relative overflow-hidden" 
             style="animation-delay: ${index * 0.1}s; box-shadow: inset 0 1px 1px rgba(255,255,255,0.05)"
             onclick="window.location.href='/details?title=${encodeURIComponent(movie.title)}'">
            
            <!-- White Shine Hover -->
            <div class="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/0 to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"></div>

            <div class="flex justify-between items-start mb-8 relative z-10">
                <h3 class="text-xl font-black text-white group-hover:text-white transition-colors leading-tight line-clamp-2 w-3/4">${movie.title}</h3>
                <div class="shrink-0 bg-zinc-900 border border-white/10 text-[10px] font-black px-3 py-1.5 rounded-full text-white tracking-widest group-hover:bg-white group-hover:text-black transition-all">
                    ${Math.round(movie.similarity_score * 100)}%
                </div>
            </div>
            
            <div class="flex items-center gap-2 mb-6 relative z-10">
                <span class="text-[9px] uppercase tracking-[0.2em] font-black text-zinc-600">${movie.release_year}</span>
                <div class="w-1 h-1 rounded-full bg-zinc-800"></div>
                <span class="text-[9px] uppercase tracking-[0.2em] font-black text-zinc-600">${movie.origin}</span>
            </div>

            <p class="text-sm text-zinc-500 line-clamp-3 mb-8 font-light leading-relaxed group-hover:text-zinc-400 transition-colors relative z-10">${movie.plot}</p>
            
            <div class="relative z-10 flex items-center justify-between">
                <div class="inline-flex items-center bg-zinc-900 border border-white/10 px-4 py-2 rounded-xl group-hover:border-white/30 transition-all">
                    <span class="text-[10px] text-zinc-500 font-bold group-hover:text-white transition-colors tracking-tight">✨ ${movie.explanation}</span>
                </div>
            </div>
        </div>
    `).join('');
    
    requestAnimationFrame(() => {
        resultsContainer.classList.remove('opacity-0');
    });
}

// Close Dropdown
document.addEventListener('click', (e) => {
    if (!movieInput.contains(e.target)) autocompleteResults.classList.add('hidden');
});
