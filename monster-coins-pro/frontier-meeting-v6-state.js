'use strict';

/* Read-only state bridge for Meeting V6. No credentials or control actions are exposed. */
try {
  if (typeof S !== 'undefined' && !window.S) window.S = S;
  if (typeof V4 !== 'undefined' && !window.V4) window.V4 = V4;
} catch (_) {
  // Meeting V6 remains fail-soft if a legacy dashboard state is unavailable.
}
