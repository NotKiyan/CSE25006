/* main.js — sidebar highlight + mobile menu */
(function () {
  'use strict';

  /* ── Mobile menu toggle ── */
  const hamburger = document.querySelector('.hamburger');
  const mobileNav = document.querySelector('.mobile-nav');
  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', () => {
      mobileNav.classList.toggle('open');
      const expanded = mobileNav.classList.contains('open');
      hamburger.setAttribute('aria-expanded', String(expanded));
    });
    /* Close on nav link click */
    mobileNav.querySelectorAll('a').forEach(a =>
      a.addEventListener('click', () => mobileNav.classList.remove('open'))
    );
  }

  /* ── Active section highlight (sidebar + mobile nav) ── */
  const sections = document.querySelectorAll('.doc-section[id]');
  const allLinks = document.querySelectorAll('.nav-list a');

  if ('IntersectionObserver' in window && sections.length) {
    const setActive = (id) => {
      allLinks.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === '#' + id);
      });
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) setActive(entry.target.id);
        });
      },
      { rootMargin: '-10% 0px -80% 0px', threshold: 0 }
    );
    sections.forEach(s => observer.observe(s));
  }
})();
