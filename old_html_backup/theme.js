// Occupant Index - Theme Toggle

(function() {
    const THEME_KEY = 'occupant-theme';

    // Get saved theme or default to dark
    function getSavedTheme() {
        return localStorage.getItem(THEME_KEY) || 'dark';
    }

    // Apply theme to document
    function applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        localStorage.setItem(THEME_KEY, theme);
        updateToggleButton(theme);
    }

    // Update toggle button text
    function updateToggleButton(theme) {
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.textContent = theme === 'dark' ? 'Light' : 'Dark';
        }
    }

    // Toggle between themes
    function toggleTheme() {
        const current = getSavedTheme();
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
    }

    // Initialize on page load
    function init() {
        // Apply saved theme immediately
        applyTheme(getSavedTheme());

        // Set up toggle button
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.addEventListener('click', toggleTheme);
        }
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Also apply theme immediately to prevent flash
    applyTheme(getSavedTheme());
})();
