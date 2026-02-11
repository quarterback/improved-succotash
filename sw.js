const CACHE_NAME = 'occupant-v1';
const STATIC_CACHE = 'occupant-static-v1';
const DATA_CACHE = 'occupant-data-v1';

// Files to cache immediately on install
const STATIC_FILES = [
  '/',
  '/index.html',
  '/cpi-data.html',
  '/calculator.html',
  '/sabermetrics.html',
  '/gov.html',
  '/about.html',
  '/styles.css',
  '/theme.js',
  '/logo.svg',
  '/manifest.json'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('[ServiceWorker] Caching static files');
      return cache.addAll(STATIC_FILES);
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== DATA_CACHE) {
            console.log('[ServiceWorker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle API/data requests differently
  if (url.pathname.startsWith('/data/')) {
    event.respondWith(
      caches.open(DATA_CACHE).then((cache) => {
        return fetch(request)
          .then((response) => {
            // Cache fresh data
            if (response.status === 200) {
              cache.put(request, response.clone());
            }
            return response;
          })
          .catch(() => {
            // Fallback to cached data if offline
            return cache.match(request);
          });
      })
    );
    return;
  }

  // Handle static files with cache-first strategy
  event.respondWith(
    caches.match(request).then((response) => {
      if (response) {
        return response;
      }

      return fetch(request).then((response) => {
        // Don't cache non-successful responses
        if (!response || response.status !== 200 || response.type === 'error') {
          return response;
        }

        // Clone the response
        const responseToCache = response.clone();

        caches.open(STATIC_CACHE).then((cache) => {
          cache.put(request, responseToCache);
        });

        return response;
      });
    })
  );
});

// Background sync for when connection is restored
self.addEventListener('sync', (event) => {
  console.log('[ServiceWorker] Background sync:', event.tag);
  if (event.tag === 'sync-data') {
    event.waitUntil(
      // Sync data when back online
      fetch('/data/compute-cpi.json')
        .then((response) => response.json())
        .then((data) => {
          console.log('[ServiceWorker] Data synced:', data);
        })
    );
  }
});

// Push notification support (for future use)
self.addEventListener('push', (event) => {
  console.log('[ServiceWorker] Push notification received');
  const options = {
    body: event.data ? event.data.text() : 'New CPI data available',
    icon: '/logo.svg',
    badge: '/logo.svg',
    vibrate: [200, 100, 200],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Data',
        icon: '/logo.svg'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/logo.svg'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Occupant Index Update', options)
  );
});
