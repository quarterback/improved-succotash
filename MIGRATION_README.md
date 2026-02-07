# React Frontend Migration

This repository has been successfully migrated from static HTML/CSS/JS to a modern React/TypeScript frontend while preserving all Python backend functionality.

## Quick Start

### Development
```bash
npm install
npm run dev
```

The dev server will start on http://localhost:5173 (or http://localhost:8080)

### Production Build
```bash
npm run build
npm run preview
```

## Project Structure

```
improved-succotash/
├── src/                    # React application source
│   ├── components/         # React components (UI library + custom)
│   ├── pages/             # Page components
│   ├── services/          # API/data services
│   ├── hooks/             # Custom React hooks
│   └── lib/               # Utilities
├── public/                # Static assets
│   └── data/              # JSON data files (served at runtime)
├── python_src/            # Python backend modules
├── data/                  # Source data (copied to public/data/)
├── compute_cpi.py         # Main Python script
└── dist/                  # Production build output (gitignored)
```

## Technology Stack

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **React Router** - Client-side routing
- **TanStack Query** - Data fetching and caching
- **Framer Motion** - Animations

### Backend (Preserved)
- **Python** - Data collection and processing
- All scripts in `python_src/` directory
- `compute_cpi.py` - Main CPI calculation script

## Available Routes

- `/` - Index page
- `/compute-cpi` - Historical CPI data dashboard
- `/calculator` - Cost calculator
- `/market-intel` - Market intelligence and rankings
- `/gov` - Government benchmarks (old HTML)
- `/about` - About page
- `/methodology` - Methodology documentation
- `/memos` - Research memos
- `/work` - Advisory services
- `/contact` - Contact page

## Data Flow

1. Python scripts in `python_src/` generate data files
2. Data files are stored in `data/` directory
3. Build process copies `data/` to `public/data/`
4. React app fetches data from `/data/*.json` at runtime
5. Services in `src/services/cpiData.ts` handle data fetching

## Python Backend

All Python functionality is preserved:

```bash
# Run Python scripts as before
python compute_cpi.py

# Python modules are in python_src/
python -m python_src.calculate_cpi
```

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm test` - Run tests (Vitest)

## Migration Notes

### What Was Changed
- Frontend: Complete rewrite in React/TypeScript
- Build system: Now using Vite instead of no build system
- Styling: Migrated to Tailwind CSS
- Old HTML files: Backed up to `old_html_backup/` directory
- Added SPA routing configuration (`public/_redirects` for Netlify, `vercel.json` for Vercel)

### What Was Preserved
- All Python backend code
- All data files and directory structure
- All data generation scripts
- Core functionality and data flow

### Data Adapters
Some data structure differences were handled with adapters in the page components:
- `Index.tsx` - Maps subindices and spreads
- `ComputeCPI.tsx` - Handles basket_detail transformation
- `MarketIntel.tsx` - Transforms rankings data

## Deployment

This is a Single Page Application (SPA) and requires proper routing configuration:

### Netlify
The `public/_redirects` file is automatically included in the build and handles SPA routing.

### Vercel
The `vercel.json` configuration file handles SPA routing.

### GitHub Pages
Requires additional configuration for SPA routing (e.g., using a 404.html trick or custom actions).

### Static Hosting
Ensure your web server is configured to serve `index.html` for all routes (not just `/`).

## Troubleshooting

### Dev server won't start
```bash
rm -rf node_modules package-lock.json
npm install
```

### Data not loading
Ensure `data/` directory is copied to `public/data/`:
```bash
cp -r data public/
```

### Python scripts failing
Python scripts expect modules to be in `python_src/`:
```bash
# Update imports if needed
from python_src.module_name import function
```

### Deploy preview shows nothing / blank page
- Check that the deployment platform is configured for SPA routing
- Verify the build output includes `_redirects` (Netlify) or uses `vercel.json` (Vercel)
- Check browser console for JavaScript errors
- Ensure the base URL is configured correctly

## Contributing

This is a migrated codebase. When making changes:
1. Frontend changes go in `src/`
2. Python changes go in `python_src/` or root for `compute_cpi.py`
3. Data schema changes may require updating adapters in page components
4. Run `npm run build` to verify production builds work

## License

© 2026 Occupant. All rights reserved.
