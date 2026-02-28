/**
 * Local-DDx Site Scripts
 * Scroll-aware nav, mobile menu toggle, active section highlighting.
 */

document.addEventListener('DOMContentLoaded', () => {

  // Scroll-aware navigation
  const nav = document.getElementById('site-nav');

  const onScroll = () => {
    nav.classList.toggle('scrolled', window.scrollY > 50);
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Mobile navigation toggle
  const toggle = document.getElementById('nav-toggle');
  const links = document.getElementById('nav-links');

  toggle.addEventListener('click', () => {
    const isOpen = links.classList.toggle('active');
    toggle.classList.toggle('active');
    toggle.setAttribute('aria-expanded', isOpen);
  });

  // Close menu when a nav link is clicked
  links.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      links.classList.remove('active');
      toggle.classList.remove('active');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });

  // Active nav link highlighting via IntersectionObserver
  const sections = document.querySelectorAll('section[id], header[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach(link => {
          link.classList.toggle('active',
            link.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, {
    rootMargin: '-64px 0px -40% 0px',
    threshold: 0
  });

  sections.forEach(section => observer.observe(section));

});
