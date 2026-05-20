// This site no longer uses a service worker. This kill-switch worker
// replaces the previous caching worker: when a returning visitor's browser
// fetches this updated script, it deletes all caches, unregisters itself,
// and reloads any controlled pages so they load fresh from the network.

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));

      await self.registration.unregister();

      const clients = await self.clients.matchAll({ type: 'window' });
      clients.forEach((client) => client.navigate(client.url));
    })()
  );
});
