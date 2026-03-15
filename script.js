
  (() => {
    document.querySelectorAll('.standalone-results-table').forEach((block) => {
      const pages = Array.from(block.querySelectorAll('.standalone-table-page'));
      const tabs = Array.from(block.querySelectorAll('.table-switcher-tab'));
      let index = 0;
      const render = () => {
        pages.forEach((page, i) => { page.hidden = i !== index; });
        tabs.forEach((tab, i) => tab.classList.toggle('active', i === index));
      };
      block.querySelectorAll('[data-table-target]').forEach((button) => {
        button.addEventListener('click', () => {
          index = Number(button.getAttribute('data-table-target') || 0);
          render();
        });
      });
      block.querySelectorAll('[data-table-nav]').forEach((button) => {
        button.addEventListener('click', () => {
          const delta = Number(button.getAttribute('data-table-nav') || 0);
          index = (index + delta + pages.length) % pages.length;
          render();
        });
      });
      render();
    });

    document.querySelectorAll('.standalone-carousel').forEach((carousel) => {
      const slides = Array.from(carousel.querySelectorAll('.standalone-carousel-slide'));
      const indicator = carousel.querySelector('.carousel-indicator');
      let index = 0;
      const render = () => {
        slides.forEach((slide, i) => { slide.hidden = i !== index; });
        if (indicator) indicator.textContent = (index + 1) + ' / ' + slides.length;
      };
      carousel.querySelectorAll('[data-carousel-nav]').forEach((button) => {
        button.addEventListener('click', () => {
          const delta = Number(button.getAttribute('data-carousel-nav') || 0);
          index = (index + delta + slides.length) % slides.length;
          render();
        });
      });
      render();
    });

    document.querySelectorAll('.comparison-container').forEach((container) => {
      const overlay = container.querySelector('.comparison-overlay');
      const line = container.querySelector('.comparison-slider-line');
      if (!overlay || !line) return;
      let dragging = false;
      const setPosition = (clientX) => {
        const rect = container.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const pct = (x / rect.width) * 100;
        overlay.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
        line.style.left = pct + '%';
      };
      const onMove = (event) => {
        if (!dragging) return;
        const point = event.touches ? event.touches[0] : event;
        setPosition(point.clientX);
      };
      const onEnd = () => { dragging = false; };
      line.addEventListener('mousedown', () => { dragging = true; });
      line.addEventListener('touchstart', () => { dragging = true; }, { passive: true });
      window.addEventListener('mousemove', onMove);
      window.addEventListener('touchmove', onMove, { passive: true });
      window.addEventListener('mouseup', onEnd);
      window.addEventListener('touchend', onEnd);
      container.addEventListener('click', (event) => setPosition(event.clientX));
    });

    // Sidebar nav: scroll spy + smooth scroll
    document.querySelectorAll('.sec-nav-sidebar').forEach((nav) => {
      const links = Array.from(nav.querySelectorAll('.sidebar-nav-item'));
      const ids = links.map(a => a.getAttribute('href')?.replace('#', '')).filter(Boolean);
      const targets = ids.map(id => document.getElementById(id)).filter(Boolean);
      if (!targets.length) return;

      const accentColor = links[0]?.style?.getPropertyValue('--nav-accent') ||
        (nav.querySelector('.sidebar-nav-dot')?.style?.background) || '#22869A';
      let activeIdx = 0;

      const setActive = (idx) => {
        activeIdx = idx;
        links.forEach((link, i) => {
          link.classList.toggle('active', i === idx);
          link.style.fontWeight = i === idx ? '700' : '500';
        });
        // Update dots track indicator
        const track = nav.querySelector('[data-track-indicator]');
        if (track) track.style.top = (idx * 38) + 'px';
      };

      // Intersection observer for scroll spy
      const observer = new IntersectionObserver((entries) => {
        let topIdx = -1, topY = Infinity;
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const idx = targets.indexOf(entry.target);
            if (idx >= 0 && entry.boundingClientRect.top < topY) {
              topY = entry.boundingClientRect.top;
              topIdx = idx;
            }
          }
        });
        if (topIdx >= 0) setActive(topIdx);
      }, { rootMargin: '-10% 0px -60% 0px', threshold: 0 });

      targets.forEach(el => observer.observe(el));

      // Smooth scroll on click
      links.forEach((link, i) => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          setActive(i);
          targets[i]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
    });
  })();
  