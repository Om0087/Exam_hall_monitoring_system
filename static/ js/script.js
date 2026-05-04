// Enhanced animations and interactions
document.addEventListener('DOMContentLoaded', function() {
    // Initialize glass morphism effects
    initGlassMorphism();

    // Enhanced ripple effect for option cards
    const optionCards = document.querySelectorAll('.option-card');

    optionCards.forEach(card => {
        card.addEventListener('click', function(e) {
            createRippleEffect(this, e);

            // Add click animation
            this.style.animation = 'pulse 0.6s ease';
            setTimeout(() => {
                this.style.animation = '';
            }, 600);
        });

        // Hover glow effect
        card.addEventListener('mouseenter', function() {
            this.style.animation = 'glow 2s infinite';
        });

        card.addEventListener('mouseleave', function() {
            this.style.animation = '';
        });
    });

    // Enhanced feature animations
    const features = document.querySelectorAll('.feature');

    features.forEach(feature => {
        feature.addEventListener('mouseenter', function() {
            this.style.animation = 'pulse 0.8s ease';
            const icon = this.querySelector('i');
            icon.style.transform = 'scale(1.3) rotate(10deg)';
        });

        feature.addEventListener('mouseleave', function() {
            this.style.animation = '';
            const icon = this.querySelector('i');
            icon.style.transform = '';
        });
    });

    // Button hover effects
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px)';
        });

        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Form input enhancements
    const formInputs = document.querySelectorAll('.form-control');
    formInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function() {
            if (this.value === '') {
                this.parentElement.classList.remove('focused');
            }
        });
    });
});

// Enhanced ripple effect function
function createRippleEffect(element, event) {
    const rect = element.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const ripple = document.createElement('span');
    ripple.className = 'ripple-effect';
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;

    element.appendChild(ripple);

    setTimeout(() => {
        ripple.remove();
    }, 800);
}

// Initialize glass morphism components
function initGlassMorphism() {
    // Add loading animation to glass cards
    const glassCards = document.querySelectorAll('.glass-card');
    glassCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';

        setTimeout(() => {
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Enhanced notification system
window.showNotification = function(message, type = 'info') {
    const notification = document.getElementById('notification');
    const content = notification.querySelector('.notification-content');

    // Create icon based on type
    let icon = '';
    switch(type) {
        case 'success':
            icon = '<i class="fas fa-check-circle"></i>';
            break;
        case 'warning':
            icon = '<i class="fas fa-exclamation-triangle"></i>';
            break;
        case 'danger':
            icon = '<i class="fas fa-times-circle"></i>';
            break;
        default:
            icon = '<i class="fas fa-info-circle"></i>';
    }

    content.innerHTML = `${icon} ${message}`;
    notification.className = `floating-notification ${type} show`;

    // Auto hide after 4 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.className = "floating-notification";
        }, 500);
    }, 4000);
};

// Utility function for smooth scrolling
window.smoothScrollTo = function(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
};

// Debounce function for performance
window.debounce = function(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    // Mobile navigation toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });

        // Close mobile menu when clicking on a link
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.nav-container')) {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            }
        });
    }

    // Active page highlighting
    const currentPage = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        const linkHref = link.getAttribute('href');
        if (currentPage.includes(linkHref.replace('/', '')) ||
            (currentPage === '/' && linkHref.includes('index'))) {
            link.classList.add('active');
        }
    });
});
