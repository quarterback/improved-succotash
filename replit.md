# Occupant Index

## Overview
Occupant is a decision infrastructure platform for AI systems. It provides real-time benchmarking and economic measurement of the AI sector, tracking the unit costs and activity volumes of AI models.

## Key Features
- **Compute CPI**: Tracks deflation/inflation of AI work costs across model tiers
- **AEAI (AI Economic Activity Index)**: Measures global AI economic activity via the AIU synthetic unit
- **Market Intelligence**: Rankings, pricing data, and model metadata
- **PWA Support**: Service worker for offline capabilities

## Tech Stack
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no build framework)
- **Backend/Data**: Python 3 scripts for data collection and index calculation
- **Data Storage**: Flat-file JSON in `data/` directory

## Project Structure
```
.
├── index.html          # Main dashboard
├── aeai.html           # Activity Index dashboard
├── cpi-data.html       # Price Index dashboard
├── displacement.html   # Labor Displacement Index (LDI)
├── calculator.html     # AI cost calculator
├── about.html          # About page
├── services.html       # Services page
├── gov.html            # Government resources
├── styles.css          # Main stylesheet
├── theme.js            # Dark/light theme toggle
├── sw.js               # PWA service worker
├── data/               # JSON data files
│   ├── aeai/           # Activity Index snapshots
│   ├── market/         # Market intelligence
│   ├── models/         # Model tier registries
│   ├── prices/         # Pricing data
│   └── rankings/       # Token volume rankings
├── src/                # Python data pipeline scripts
│   ├── calculate_aeai.py
│   ├── calculate_cpi.py
│   ├── data_collector.py
│   ├── model_registry.py
│   └── scrape_rankings.py
└── fonts/              # Custom brand fonts
```

## Running the App
The app is served as a static site using Python's built-in HTTP server on port 5000.

**Workflow**: `Start application`
**Command**: `python3 -m http.server 5000 --bind 0.0.0.0`

## Deployment
Configured as a static site deployment with the root directory (`.`) as the public directory.
