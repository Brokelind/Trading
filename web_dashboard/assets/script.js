
// Auto-refresh dashboard every 5 minutes
setTimeout(() => {
    window.location.reload();
}, 300000);
// Add smooth animations
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

// Add to existing CSS:
// .fade-in { animation: fadeIn 0.5s ease-out forwards; opacity: 0; }
// @keyframes fadeIn { to { opacity: 1; } }
