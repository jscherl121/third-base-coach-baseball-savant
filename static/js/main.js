/**
 * Third Base Send Model - Main JavaScript
 */

// Smooth animations on scroll
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to elements as they come into view
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards and other elements
    document.querySelectorAll('.card, .play-card').forEach(el => {
        observer.observe(el);
    });
});

// Tooltip functionality
document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.classList.add('tooltip');
});

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format percentage
function formatPercent(num, decimals = 1) {
    return (num * 100).toFixed(decimals) + '%';
}

// Color scale for WP values
function getWPColor(value) {
    if (value > 0.05) return 'var(--success)';
    if (value > 0) return '#8bc34a';
    if (value > -0.05) return 'var(--warning)';
    return 'var(--danger)';
}

// Team logo mapping (could be extended with actual logos)
const teamColors = {
    'AZ': '#A71930',
    'ATL': '#CE1141',
    'BAL': '#DF4601',
    'BOS': '#BD3039',
    'CHC': '#0E3386',
    'CWS': '#27251F',
    'CIN': '#C6011F',
    'CLE': '#00385D',
    'COL': '#33006F',
    'DET': '#0C2340',
    'HOU': '#002D62',
    'KC': '#004687',
    'LAA': '#BA0021',
    'LAD': '#005A9C',
    'MIA': '#00A3E0',
    'MIL': '#12284B',
    'MIN': '#002B5C',
    'NYM': '#002D72',
    'NYY': '#003087',
    'ATH': '#003831',
    'PHI': '#E81828',
    'PIT': '#FDB827',
    'SD': '#2F241D',
    'SF': '#FD5A1E',
    'SEA': '#0C2C56',
    'STL': '#C41E3A',
    'TB': '#092C5C',
    'TEX': '#003278',
    'TOR': '#134A8E',
    'WSH': '#AB0003'
};

// Apply team colors to team abbreviations
document.querySelectorAll('.team-abbr').forEach(el => {
    const team = el.textContent.trim();
    if (teamColors[team]) {
        el.style.backgroundColor = teamColors[team];
        el.style.color = '#ffffff';
    }
});

// Keyboard navigation for play cards
document.querySelectorAll('.play-card').forEach(card => {
    card.setAttribute('tabindex', '0');
    card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            card.click();
        }
    });
});

// Search functionality (if search is added later)
function searchPlays(query) {
    const cards = document.querySelectorAll('.play-card');
    const normalizedQuery = query.toLowerCase().trim();

    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        if (text.includes(normalizedQuery) || normalizedQuery === '') {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

// Export data functionality
function exportTableToCSV(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const rows = table.querySelectorAll('tr');
    const csv = [];

    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        const rowData = [];
        cols.forEach(col => {
            // Clean the text content
            let text = col.textContent.trim().replace(/"/g, '""');
            rowData.push(`"${text}"`);
        });
        csv.push(rowData.join(','));
    });

    const csvContent = csv.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

// Print functionality
function printPage() {
    window.print();
}

// Console easter egg
console.log('%c⚾ Third Base Send Model', 'font-size: 24px; font-weight: bold; color: #e94560;');
console.log('%cEvaluating third base coach decisions using Win Probability analysis.', 'color: #00adb5;');
console.log('%cBuilt with Statcast data from Baseball Savant.', 'color: #888;');
