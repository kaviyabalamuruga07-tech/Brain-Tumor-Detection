// ================================================================
//  NeuroScan V3.0 — main.js
// ================================================================

// ── NAVBAR SCROLL ────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (nav) {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  }

  // Highlight active nav link based on scroll position
  const sections = ['home','features','scan','about','contact'];
  let current = '';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 100) current = id;
  });
  document.querySelectorAll('.nav-menu a').forEach(a => {
    a.classList.remove('active');
    if (a.getAttribute('href') === '#' + current) a.classList.add('active');
  });
});

// ── MOBILE MENU ──────────────────────────────────────────────
function toggleMenu() {
  const menu = document.getElementById('navMenu');
  if (menu) menu.classList.toggle('open');
}

// Close menu when a link is clicked
document.querySelectorAll('.nav-menu a').forEach(a => {
  a.addEventListener('click', () => {
    const menu = document.getElementById('navMenu');
    if (menu) menu.classList.remove('open');
  });
});

// ── IMAGE PICK ───────────────────────────────────────────────
function onPick(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const reader = new FileReader();

  reader.onload = (e) => {
    document.getElementById('dzContent').style.display = 'none';
    const preview = document.getElementById('preview');
    preview.style.display = 'block';
    document.getElementById('pimg').src = e.target.result;
    document.getElementById('pname').textContent = '📎 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── DRAG & DROP ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const dz = document.getElementById('dz');
  if (dz) {
    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      dz.classList.add('dragover');
    });
    dz.addEventListener('dragleave', () => {
      dz.classList.remove('dragover');
    });
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) {
        const fi = document.getElementById('fi');
        const dt = new DataTransfer();
        dt.items.add(file);
        fi.files = dt.files;
        onPick(fi);
      }
    });
  }

  // Form submit loading state
  const form = document.getElementById('form');
  if (form) {
    form.addEventListener('submit', () => {
      const btn = document.getElementById('btn');
      const txt = document.getElementById('btnText');
      if (btn) btn.disabled = true;
      if (txt) txt.textContent = 'Analyzing... Please wait ⏳';
    });
  }
});

// ── SMOOTH SCROLL ────────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', (e) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── CONTACT FORM ─────────────────────────────────────────────
function sendMsg(e) {
  e.preventDefault();
  const btn = e.target;
  btn.textContent = '⏳ Sending...';
  btn.disabled = true;
  setTimeout(() => {
    btn.textContent = '📤 Send Message';
    btn.disabled = false;
    const msg = document.getElementById('successMsg');
    if (msg) {
      msg.style.display = 'block';
      setTimeout(() => { msg.style.display = 'none'; }, 4000);
    }
  }, 1500);
}

// ── ANIMATE ON SCROLL ────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.feat-card, .how-step, .contact-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});
