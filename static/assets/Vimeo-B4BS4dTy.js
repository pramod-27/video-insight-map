import{r as L,a as S,b as R,g as k}from"./index-DqD9V3by.js";function T(p,a){for(var u=0;u<a.length;u++){const n=a[u];if(typeof n!="string"&&!Array.isArray(n)){for(const l in n)if(l!=="default"&&!(l in p)){const h=Object.getOwnPropertyDescriptor(n,l);h&&Object.defineProperty(p,l,h.get?h:{enumerable:!0,get:()=>n[l]})}}}return Object.freeze(Object.defineProperty(p,Symbol.toStringTag,{value:"Module"}))}var d,g;function q(){if(g)return d;g=1;var p=Object.create,a=Object.defineProperty,u=Object.getOwnPropertyDescriptor,n=Object.getOwnPropertyNames,l=Object.getPrototypeOf,h=Object.prototype.hasOwnProperty,v=(t,e,r)=>e in t?a(t,e,{enumerable:!0,configurable:!0,writable:!0,value:r}):t[e]=r,O=(t,e)=>{for(var r in e)a(t,r,{get:e[r],enumerable:!0})},f=(t,e,r,y)=>{if(e&&typeof e=="object"||typeof e=="function")for(let i of n(e))!h.call(t,i)&&i!==r&&a(t,i,{get:()=>e[i],enumerable:!(y=u(e,i))||y.enumerable});return t},D=(t,e,r)=>(r=t!=null?p(l(t)):{},f(!t||!t.__esModule?a(r,"default",{value:t,enumerable:!0}):r,t)),V=t=>f(a({},"__esModule",{value:!0}),t),s=(t,e,r)=>(v(t,typeof e!="symbol"?e+"":e,r),r),m={};O(m,{default:()=>c}),d=V(m);var _=D(L()),P=S(),w=R();const M="https://player.vimeo.com/api/player.js",j="Vimeo",E=t=>t.replace("/manage/videos","");class c extends _.Component{constructor(){super(...arguments),s(this,"callPlayer",P.callPlayer),s(this,"duration",null),s(this,"currentTime",null),s(this,"secondsLoaded",null),s(this,"mute",()=>{this.setMuted(!0)}),s(this,"unmute",()=>{this.setMuted(!1)}),s(this,"ref",e=>{this.container=e})}componentDidMount(){this.props.onMount&&this.props.onMount(this)}load(e){this.duration=null,(0,P.getSDK)(M,j).then(r=>{if(!this.container)return;const{playerOptions:y,title:i}=this.props.config;this.player=new r.Player(this.container,{url:E(e),autoplay:this.props.playing,muted:this.props.muted,loop:this.props.loop,playsinline:this.props.playsinline,controls:this.props.controls,...y}),this.player.ready().then(()=>{const o=this.container.querySelector("iframe");o.style.width="100%",o.style.height="100%",i&&(o.title=i)}).catch(this.props.onError),this.player.on("loaded",()=>{this.props.onReady(),this.refreshDuration()}),this.player.on("play",()=>{this.props.onPlay(),this.refreshDuration()}),this.player.on("pause",this.props.onPause),this.player.on("seeked",o=>this.props.onSeek(o.seconds)),this.player.on("ended",this.props.onEnded),this.player.on("error",this.props.onError),this.player.on("timeupdate",({seconds:o})=>{this.currentTime=o}),this.player.on("progress",({seconds:o})=>{this.secondsLoaded=o}),this.player.on("bufferstart",this.props.onBuffer),this.player.on("bufferend",this.props.onBufferEnd),this.player.on("playbackratechange",o=>this.props.onPlaybackRateChange(o.playbackRate))},this.props.onError)}refreshDuration(){this.player.getDuration().then(e=>{this.duration=e})}play(){const e=this.callPlayer("play");e&&e.catch(this.props.onError)}pause(){this.callPlayer("pause")}stop(){this.callPlayer("unload")}seekTo(e,r=!0){this.callPlayer("setCurrentTime",e),r||this.pause()}setVolume(e){this.callPlayer("setVolume",e)}setMuted(e){this.callPlayer("setMuted",e)}setLoop(e){this.callPlayer("setLoop",e)}setPlaybackRate(e){this.callPlayer("setPlaybackRate",e)}getDuration(){return this.duration}getCurrentTime(){return this.currentTime}getSecondsLoaded(){return this.secondsLoaded}render(){const{display:e}=this.props,r={width:"100%",height:"100%",overflow:"hidden",display:e};return _.default.createElement("div",{key:this.props.url,ref:this.ref,style:r})}}return s(c,"displayName","Vimeo"),s(c,"canPlay",w.canPlay.vimeo),s(c,"forceLoad",!0),d}var b=q();const x=k(b),N=T({__proto__:null,default:x},[b]);export{N as V};
