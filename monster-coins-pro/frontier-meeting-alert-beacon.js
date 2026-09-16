'use strict';

/* Visual-only warning beacon for the existing meeting button. No meeting/decision logic changes. */
(function installMeetingAlertBeacon(){
  if(document.getElementById('frontierMeetingAlertBeaconStyle')) return;
  const style=document.createElement('style');
  style.id='frontierMeetingAlertBeaconStyle';
  style.textContent=`
    #meetingCard #openMeeting{position:relative;isolation:isolate;overflow:visible!important;transition:box-shadow .2s ease,border-color .2s ease,transform .2s ease}
    #meetingCard.meeting-v6-red #openMeeting{z-index:1;border-color:rgba(255,63,86,.96)!important;box-shadow:0 0 0 1px rgba(255,45,70,.42),0 0 16px rgba(255,34,60,.58),0 0 36px rgba(255,24,48,.34);animation:meetingCriticalButtonAlarm .82s ease-in-out infinite}
    #meetingCard.meeting-v6-red #openMeeting:before,#meetingCard.meeting-v6-red #openMeeting:after{content:"";position:absolute;pointer-events:none;border-radius:inherit;inset:-4px;z-index:-1;border:2px solid rgba(255,54,78,.72);animation:meetingCriticalButtonHalo .82s ease-out infinite}
    #meetingCard.meeting-v6-red #openMeeting:after{inset:-10px;border-width:1px;animation-delay:.28s}
    #meetingCard.meeting-v6-orange #openMeeting{z-index:1;border-color:rgba(255,161,48,.92)!important;box-shadow:0 0 0 1px rgba(255,151,37,.32),0 0 14px rgba(255,139,28,.48),0 0 30px rgba(255,118,18,.25);animation:meetingHighButtonAlarm 1.25s ease-in-out infinite}
    #meetingCard.meeting-v6-orange #openMeeting:before,#meetingCard.meeting-v6-orange #openMeeting:after{content:"";position:absolute;pointer-events:none;border-radius:inherit;inset:-4px;z-index:-1;border:2px solid rgba(255,164,53,.58);animation:meetingHighButtonHalo 1.25s ease-out infinite}
    #meetingCard.meeting-v6-orange #openMeeting:after{inset:-9px;border-width:1px;animation-delay:.38s}
    @keyframes meetingCriticalButtonAlarm{0%,100%{box-shadow:0 0 0 1px rgba(255,45,70,.30),0 0 9px rgba(255,34,60,.38),0 0 22px rgba(255,24,48,.18);transform:translateZ(0) scale(1)}50%{box-shadow:0 0 0 2px rgba(255,73,94,.72),0 0 25px rgba(255,45,69,.92),0 0 58px rgba(255,25,52,.55);transform:translateZ(0) scale(1.018)}}
    @keyframes meetingCriticalButtonHalo{0%{transform:scale(.98);opacity:.92}100%{transform:scale(1.10);opacity:0}}
    @keyframes meetingHighButtonAlarm{0%,100%{box-shadow:0 0 0 1px rgba(255,151,37,.24),0 0 8px rgba(255,139,28,.30),0 0 18px rgba(255,118,18,.14);transform:translateZ(0) scale(1)}50%{box-shadow:0 0 0 2px rgba(255,180,75,.58),0 0 21px rgba(255,158,45,.80),0 0 48px rgba(255,126,22,.40);transform:translateZ(0) scale(1.012)}}
    @keyframes meetingHighButtonHalo{0%{transform:scale(.985);opacity:.78}100%{transform:scale(1.085);opacity:0}}
    @media (prefers-reduced-motion:reduce){#meetingCard.meeting-v6-red #openMeeting,#meetingCard.meeting-v6-orange #openMeeting,#meetingCard.meeting-v6-red #openMeeting:before,#meetingCard.meeting-v6-red #openMeeting:after,#meetingCard.meeting-v6-orange #openMeeting:before,#meetingCard.meeting-v6-orange #openMeeting:after{animation:none!important}}
  `;
  document.head.appendChild(style);
})();
