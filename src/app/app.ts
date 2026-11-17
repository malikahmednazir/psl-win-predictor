import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { DecimalPipe } from '@angular/common';

type Row = Record<string, string>;

const COLORS = ['#34d399', '#60a5fa', '#fbbf24', '#f472b6', '#a78bfa', '#f87171'];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  summary: Row[] = [];
  e1: Row[] = [];
  e2: Row[] = [];
  e3: Row[] = [];
  e3ipl: Row[] = [];

  loading = true;
  error = '';

  // UI state
  e1Phase = '1-20';
  e2League = 'PSL';
  e2Split = '';
  e3Model = 'CatBoost';

  readonly metrics = [
    { label: 'Accuracy', key: 'Accuracy' },
    { label: 'Precision', key: 'Precision' },
    { label: 'Recall', key: 'Recall' },
    { label: 'F1-score', key: 'F1' },
    { label: 'ROC-AUC', key: 'ROC_AUC' }
  ];
  readonly phases = ['1-20', '1-5', '6-10', '11-15', '16-20'];

  constructor(private cdr: ChangeDetectorRef) {}

  async ngOnInit(): Promise<void> {
    const base = document.querySelector('base')?.href ?? window.location.href;
    const get = async (file: string): Promise<Row[]> => {
      try {
        const r = await fetch(new URL('research-data/' + file, base));
        return r.ok ? this.parseCsv(await r.text()) : [];
      } catch {
        return [];
      }
    };
    [this.summary, this.e1, this.e2, this.e3, this.e3ipl] = await Promise.all([
      get('dataset_summary.csv'),
      get('e1_paper_reproduction_ipl.csv'),
      get('e2_four_models.csv'),
      get('e3_novelty_psl_rolling.csv'),
      get('e3_novelty_ipl2019_features.csv')
    ]);
    if (!this.summary.length && !this.e2.length) {
      this.error = 'Could not load research results. Run ml/run_experiments.py and refresh.';
    }
    this.e2Split = this.splitsFor(this.e2League)[0] ?? '';
    this.loading = false;
    this.cdr.detectChanges();
  }

  private parseCsv(csv: string): Row[] {
    const text = csv.replace(/^\uFEFF/, '').trim();

    if (!text) {
      return [];
    }

    const rows: string[][] = [];
    let row: string[] = [];
    let field = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      if (char === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        row.push(field.trim());
        field = '';
      } else if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && text[i + 1] === '\n') {
          i++;
        }

        row.push(field.trim());

        if (row.some(value => value !== '')) {
          rows.push(row);
        }

        row = [];
        field = '';
      } else {
        field += char;
      }
    }

    row.push(field.trim());

    if (row.some(value => value !== '')) {
      rows.push(row);
    }

    if (rows.length < 2) {
      return [];
    }

    const headers = rows[0].map(header => header.trim());

    return rows.slice(1).map(values => {
      const result: Row = {};

      headers.forEach((header, index) => {
        result[header] = (values[index] ?? '').trim();
      });

      return result;
    });
  }

  // ---------------- generic helpers
  n(row: Row | undefined, key: string): number {
    const v = Number(row?.[key]);
    return Number.isFinite(v) ? v : NaN;
  }

  f(v: number, d = 2, suffix = ''): string {
    return Number.isFinite(v) ? v.toFixed(d) + suffix : '—';
  }

  uniq(rows: Row[], key: string): string[] {
    return [...new Set(rows.map(r => r[key]))];
  }

  color(i: number): string {
    return COLORS[i % COLORS.length];
  }

  mean(xs: number[]): number {
    const v = xs.filter(Number.isFinite);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  }

  std(xs: number[]): number {
    const v = xs.filter(Number.isFinite);
    if (v.length < 2) return NaN;
    const m = this.mean(v);
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  }

  // ---------------- dataset summary
  leagueTotal(league: string, key: string): number {
    return this.summary.filter(r => r['League'] === league).reduce((a, r) => a + this.n(r, key), 0);
  }

  seasonsOf(league: string): Row[] {
    return this.summary.filter(r => r['League'] === league);
  }

  barPct(v: number, max: number): number {
    return Number.isFinite(v) && max > 0 ? Math.max(0, Math.min(100, (v / max) * 100)) : 0;
  }

  get maxSeasonMatches(): number {
    return Math.max(1, ...this.summary.map(r => this.n(r, 'Matches')));
  }

  // ---------------- E1 paper reproduction
  e1Rows(): Row[] {
    return this.e1.filter(r => r['Phase'] === this.e1Phase);
  }

  e1Diff(r: Row): number {
    return this.n(r, 'Accuracy') - this.n(r, 'Paper_test_accuracy');
  }

  e1Best(family: string): Row | undefined {
    return this.e1Rows().filter(r => r['Family'] === family)
      .sort((a, b) => this.n(b, 'Accuracy') - this.n(a, 'Accuracy'))[0];
  }

  // ---------------- E2 four models
  splitsFor(league: string): string[] {
    return this.uniq(this.e2.filter(r => r['League'] === league), 'Split');
  }

  setLeague(l: string): void {
    this.e2League = l;
    if (!this.splitsFor(l).includes(this.e2Split)) this.e2Split = this.splitsFor(l)[0] ?? '';
  }

  e2Rows(): Row[] {
    return this.e2.filter(r => r['League'] === this.e2League && r['Split'] === this.e2Split);
  }

  e2Max(key: string): number {
    return Math.max(...this.e2Rows().map(r => this.n(r, key)));
  }

  // ---------------- E3 novelty
  get e3Settings(): string[] {
    return this.uniq(this.e3, 'Setting');
  }

  get e3Models(): string[] {
    return this.uniq(this.e3, 'Model');
  }

  get e3Seasons(): string[] {
    return this.uniq(this.e3, 'Test_season');
  }

  e3Agg(setting: string, key: string, model = this.e3Model): { m: number; s: number } {
    const xs = this.e3.filter(r => r['Setting'] === setting && r['Model'] === model).map(r => this.n(r, key));
    return { m: this.mean(xs), s: this.std(xs) };
  }

  e3Cell(setting: string, season: string, key = 'Accuracy'): number {
    return this.n(this.e3.find(r => r['Setting'] === setting && r['Model'] === this.e3Model &&
      r['Test_season'] === season), key);
  }

  isProposed(s: string): boolean {
    return s.startsWith('S4');
  }

  isBaseline(s: string): boolean {
    return s.startsWith('S1');
  }

  /** Improvement of the proposed setting over the paper method, averaged over seasons, for each model */
  gain(model: string, key: string): number {
    const s1 = this.e3Settings.find(s => s.startsWith('S1'));
    const s4 = this.e3Settings.find(s => s.startsWith('S4'));
    if (!s1 || !s4) return NaN;
    return this.e3Agg(s4, key, model).m - this.e3Agg(s1, key, model).m;
  }

  winsOf(model: string): string {
    const s1 = this.e3Settings.find(s => s.startsWith('S1'));
    const s4 = this.e3Settings.find(s => s.startsWith('S4'));
    if (!s1 || !s4) return '—';
    let w = 0, t = 0;
    for (const season of this.e3Seasons) {
      const a = this.n(this.e3.find(r => r['Setting'] === s1 && r['Model'] === model && r['Test_season'] === season), 'Accuracy');
      const b = this.n(this.e3.find(r => r['Setting'] === s4 && r['Model'] === model && r['Test_season'] === season), 'Accuracy');
      if (Number.isFinite(a) && Number.isFinite(b)) { t++; if (b > a) w++; }
    }
    return `${w} / ${t}`;
  }

  heat(v: number, lo: number, hi: number): string {
    if (!Number.isFinite(v)) return 'transparent';
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    return `hsla(${t * 120}, 65%, 45%, 0.35)`;
  }

  // ---------------- E3 IPL feature check
  e3iplRows(): Row[] {
    return this.e3ipl;
  }
}
