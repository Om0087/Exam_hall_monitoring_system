// Add ripple effect to option cards
document.addEventListener('DOMContentLoaded', function() {
    const optionCards = document.querySelectorAll('.option-card');

    optionCards.forEach(card => {
        card.addEventListener('click', function(e) {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const ripple = card.querySelector('.ripple-effect');
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            ripple.style.animation = 'ripple 0.6s linear';

            setTimeout(() => {
                ripple.style.animation = 'none';
            }, 600);
        });
    });

    // Pulse animation for features
    const features = document.querySelectorAll('.feature');

    features.forEach(feature => {
        feature.addEventListener('mouseenter', function() {
            this.style.animation = 'pulse 0.5s ease';
        });

        feature.addEventListener('mouseleave', function() {
            this.style.animation = 'none';
        });
    });

    // Notification system
    window.showNotification = function(message, type) {
        const notification = document.getElementById('notification');
        const content = notification.querySelector('.notification-content');

        content.textContent = message;
        notification.className = `floating-notification ${type} show`;

        setTimeout(() => {
            notification.className = "floating-notification";
        }, 3000);
    };
});