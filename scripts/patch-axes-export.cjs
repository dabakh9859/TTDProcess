// Visualisation : titres d'axes (avec unités), légende sans pictogramme, textes
// d'axes plus grands, et export 📷 réduit au seul graphique (sans barre de zoom
// ni icônes). À appliquer après patch.js. Chaque remplacement vérifie son ancrage.
const fs = require("fs");
const [, , inPath, outPath] = process.argv;
let s = fs.readFileSync(inPath, "utf8");

function rep(a, b, count = 1) {
  const n = s.split(a).length - 1;
  if (n !== count) throw new Error(`ancrage trouvé ${n}× (attendu ${count}) : ${a.slice(0, 90)}`);
  s = s.split(a).join(b);
}

// ---------------------------------------------------------------- helpers
rep("function TTDXYLine(",
  'function TTD_src(v){return String(v).replace(/^(📊|📂) /u,"")}' +
  'function TTD_yname(txt,side){return txt?{name:txt,nameLocation:"middle",nameRotate:side?-90:90,nameGap:side?52:56,nameTextStyle:{color:"#52514e",fontSize:13}}:{}}' +
  'function TTD_axisInputs(y0,s0,y1,s1,x,sx,dual){const inp=(lab,val,set,ph)=>(0,h.jsxs)("label",{style:{display:"block",marginTop:8},children:[ttdLabel(lab),(0,h.jsx)("input",{value:val,placeholder:ph,onChange:e=>set(e.target.value),style:{width:"100%",padding:"7px 10px",borderRadius:8,border:"1px solid var(--border-2)",background:"var(--bg-2)",color:"var(--text-1)",fontSize:12,boxSizing:"border-box"}})]});' +
  'return(0,h.jsxs)("div",{style:{marginTop:10},children:[inp("Titre de l\'axe Y (gauche)",y0,s0,"ex. Flux de sève (l dm⁻² j⁻¹)"),dual&&inp("Titre de l\'axe Y droit",y1,s1,"ex. ETo (mm sem⁻¹)"),inp("Titre de l\'axe X",x,sx,"vide = automatique")]})}' +
  "function TTDXYLine(");

// ---------------------------------------------------------------- état + UI
rep("const[NRM,SETNRM]=(0,j.useState)(!1);",
  'const[NRM,SETNRM]=(0,j.useState)(!1);const[YT0,SETYT0]=(0,j.useState)(""),[YT1,SETYT1]=(0,j.useState)(""),[XT,SETXT]=(0,j.useState)("");');
rep('children:t("viz.invertYHint")}),',
  'children:t("viz.invertYHint")}),TTD_axisInputs(YT0,SETYT0,YT1,SETYT1,XT,SETXT,T),');
rep("[xt,vt,T,n,Dr,I,O,q,Z,J,t,NRM])", "[xt,vt,T,n,Dr,I,O,q,Z,J,t,NRM,YT0,YT1,XT])");

// ---------------------------------------------------------------- graphique temporel
rep("inverse:O,min:J.a0.min??void 0", "inverse:O,...TTD_yname(YT0,0),min:J.a0.min??void 0", 2);
rep("inverse:O,min:J.a1.min??void 0", "inverse:O,...TTD_yname(YT1,1),min:J.a1.min??void 0");
rep("axisLabel:{color:ge,fontSize:10,hideOverlap:!0,...(TTD_isMonth",
  '...(XT?{name:XT,nameLocation:"middle",nameGap:32,nameTextStyle:{color:"#52514e",fontSize:13}}:{}),axisLabel:{color:ge,fontSize:12,hideOverlap:!0,...(TTD_isMonth');
rep("axisLabel:{color:ge,fontSize:10,formatter:ST}", "axisLabel:{color:ge,fontSize:12,formatter:ST}", 3);
rep("mt={name:`${tt.columns[Tt]} · ${Dr(tt.source)}`", "mt={name:`${tt.columns[Tt]} · ${TTD_src(Dr(tt.source))}`");

// ---------------------------------------------------------------- Courbe XY
rep("sourceLabel:Dr,t});if(I===\"corr\")", "sourceLabel:Dr,t,yt:YT0,xt2:XT});if(I===\"corr\")");
rep("name:`${y.name} · ${a(y.source)}`", "name:`${y.name} · ${TTD_src(a(y.source))}`");
rep('xAxis:{type:"value",scale:!0,name:e[0].columns[0],',
  'xAxis:{type:"value",scale:!0,name:t.xt2||e[0].columns[0],');
rep('yAxis:{type:"value",scale:!0,axisLabel:{color:n.textColor,fontSize:10},',
  'yAxis:{type:"value",scale:!0,...TTD_yname(t.yt,0),axisLabel:{color:n.textColor,fontSize:12},');

// ---------------------------------------------------------------- export 📷
rep('X.getDataURL({pixelRatio:2,backgroundColor:"#ffffff"})',
  'X.getDataURL({pixelRatio:2,backgroundColor:"#ffffff",excludeComponents:["toolbox","dataZoom"]})');


// ---------------------------------------------------------------- légende au-dessus du graphique
rep('legend:{type:"scroll",orient:"vertical",right:12,top:46,itemWidth:20,itemHeight:10,itemGap:6,textStyle:{color:ge,fontSize:10}',
  'legend:{type:"plain",orient:"horizontal",left:"center",right:40,top:4,itemWidth:20,itemHeight:10,itemGap:14,textStyle:{color:ge,fontSize:12}');
rep('grid:{top:56,bottom:76,left:60,right:T?60:24,containLabel:!0}', 'grid:{top:St.length>8?112:St.length>4?88:64,bottom:76,left:60,right:T?60:24,containLabel:!0}');

// ---------------------------------------------------------------- axe « journée moyenne » : heures au lieu de « 1970 »
rep('function TTD_isMonth(', 'function TTD_isDay(St){return St.length>0&&St.every(_s=>_s.data.every(([_x])=>new Date(_x).getFullYear()===1970))}function TTD_isMonth(');
rep('...(TTD_isMonth(St)?{formatter:_v=>TTD_MOIS[new Date(_v).getMonth()]}:{})', '...(TTD_isMonth(St)?{formatter:_v=>TTD_MOIS[new Date(_v).getMonth()]}:TTD_isDay(St)?{formatter:_v=>new Date(_v).getHours()+" h"}:{})');

// ---------------------------------------------------------------- barres : l'axe Y part de 0 (sinon les hauteurs trompent)
rep('scale:!0,inverse:O', 'scale:!St.some(_q=>_q.type==="bar"),inverse:O', 3);

fs.writeFileSync(outPath, s);
console.log("patch2 appliqué :", outPath);
