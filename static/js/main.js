// static/js/main.js — Common helpers for DEA ChampionSys

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 6 seconds
    const flashes = document.querySelectorAll('.flash-glass');
    flashes.forEach(function(flash) {
        setTimeout(function() {
            flash.style.transition = 'opacity 300ms ease-out, transform 300ms ease-out';
            flash.style.opacity = '0';
            flash.style.transform = 'translateY(-8px)';
            setTimeout(function() { flash.remove(); }, 300);
        }, 6000);
    });

    // Navbar scroll effect
    const nav = document.querySelector('nav');
    if (nav) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 10) {
                nav.style.boxShadow = '0 12px 48px rgba(79, 70, 229, 0.12)';
            } else {
                nav.style.boxShadow = '0 8px 32px rgba(79, 70, 229, 0.08)';
            }
        });
    }

    // Animate cards on load
    const cards = document.querySelectorAll('.card-glass, .card-kpi');
    cards.forEach(function(card, index) {
        card.style.opacity = '0';
        card.style.transform = 'translateY(8px)';
        setTimeout(function() {
            card.style.transition = 'opacity 300ms ease-out, transform 300ms ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 80 * index);
    });
});

// Number formatting helper
function formatNumber(num, decimals) {
    if (decimals === undefined) decimals = 4;
    if (typeof num !== 'number') return num;
    return num.toLocaleString('es-ES', { minimumFractionDigits: 0, maximumFractionDigits: decimals });
}
