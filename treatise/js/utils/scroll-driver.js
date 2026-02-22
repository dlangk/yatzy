/**
 * ScrollDriver — triggers step callbacks via IntersectionObserver.
 */
export class ScrollDriver {
  constructor(container, steps, onStep) {
    this.container = container;
    this.steps = steps;         // [{el, enter(), leave()}]
    this.onStep = onStep;       // (idx) => void
    this.observers = [];
    this.activeIdx = -1;
  }

  start() {
    this.steps.forEach((step, idx) => {
      const obs = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              this._activate(idx);
            }
          });
        },
        { rootMargin: '-30% 0px -50% 0px', threshold: 0 }
      );
      obs.observe(step.el);
      this.observers.push(obs);
    });
  }

  _activate(idx) {
    if (idx === this.activeIdx) return;
    // Deactivate previous
    if (this.activeIdx >= 0 && this.steps[this.activeIdx].leave) {
      this.steps[this.activeIdx].leave();
    }
    this.activeIdx = idx;
    // Activate new
    if (this.steps[idx].enter) {
      this.steps[idx].enter();
    }
    if (this.onStep) this.onStep(idx);
  }

  goTo(idx) {
    if (idx >= 0 && idx < this.steps.length) {
      this._activate(idx);
    }
  }

  destroy() {
    this.observers.forEach((obs) => obs.disconnect());
    this.observers = [];
    this.activeIdx = -1;
  }
}
