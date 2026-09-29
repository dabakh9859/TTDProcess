// Ajoute à TTDProcess : Agrégation « Mois moyen » et « Par classes », bouton
// « Ouvrir dans Visualisation », graphiques « Courbe XY » et « Corrélations »,
// option « Indexer de 0 à 1 ». Chaque remplacement vérifie son ancrage.
const fs = require("fs");
const [, , inPath, outPath] = process.argv;
let s = fs.readFileSync(inPath, "utf8");

function rep(a, b, count = 1) {
  const n = s.split(a).length - 1;
  if (n !== count) throw new Error(`ancrage trouvé ${n}× (attendu ${count}) : ${a.slice(0, 90)}`);
  s = s.split(a).join(b);
}

// ---------------------------------------------------------------- i18n
rep('"agg.period.month.label":{fr:"Mois",en:"Month"}',
  '"agg.period.month.label":{fr:"Mois",en:"Month"},' +
  '"viz.chartType.xyline":{fr:"Courbe XY",en:"XY line"},' +
  '"viz.chartType.xylineHint":{fr:"1ʳᵉ série = axe horizontal (ex. centre de classe de VPD). Toutes les autres séries sont tracées en courbes, reliées dans l\'ordre de X. Idéal pour une agrégation « Par classes ».",en:"1st series = x axis (e.g. VPD class centre). Every other series is drawn as a line ordered by x. Made for a \\"By classes\\" aggregation."},' +
  '"viz.chartType.corr":{fr:"Corrélations",en:"Correlations"},' +
  '"viz.chartType.corrHint":{fr:"Corrélation de Spearman (ρ, de −1 à 1) entre toutes les colonnes visibles, appariées par date. Mets des séries au même pas de temps, par ex. l\'agrégation journalière du flux et celle de la météo. Bleu = varient ensemble, rouge = en sens opposé.",en:"Spearman correlation (ρ, −1 to 1) between every visible column, paired by date. Use series on the same time step, e.g. daily aggregations of sap flow and weather. Blue = move together, red = opposite."},' +
  '"viz.normalize":{fr:"Indexer de 0 à 1 (min → max)",en:"Index 0 to 1 (min → max)"},' +
  '"viz.normalizeHint":{fr:"Chaque courbe est ramenée entre 0 (son minimum) et 1 (son maximum) : on compare la forme et l\'horaire de variables d\'unités différentes (flux, Rn, VPD…) sans double axe.",en:"Each line is rescaled between 0 (its minimum) and 1 (its maximum) to compare the shape and timing of variables with different units, without a second axis."}');

// ---------------------------------------------------------------- Agrégation : panneaux
const MOIS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
const helpers = `
var TTD_MOIS=${JSON.stringify(["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."])};
function ttdNote(children){return(0,h.jsx)("div",{style:{fontSize:11,color:"var(--text-4)",lineHeight:1.55},children})}
function ttdLabel(txt){return(0,h.jsx)("div",{style:{fontSize:11,color:"var(--text-3)",marginBottom:6,fontWeight:500},children:txt})}
function ttdChip(active,label,onClick,key){return(0,h.jsx)("button",{onClick,style:{padding:"6px 0",borderRadius:6,fontSize:11,fontWeight:600,cursor:"pointer",background:active?"var(--accent-tint-10)":"var(--bg-2)",border:\`1px solid \${active?"var(--accent-tint-30)":"var(--border-2)"}\`,color:active?"var(--accent-hover)":"var(--text-3)"},children:label},key)}
function ttdSmallBtn(label,onClick){return(0,h.jsx)("button",{onClick,style:{padding:"4px 10px",fontSize:11,borderRadius:6,border:"1px solid var(--border-2)",background:"var(--bg-2)",color:"var(--text-2)",cursor:"pointer"},children:label})}
function TTD_isMonth(St){return St.length>0&&St.every(_s=>_s.data.every(([_x,_y])=>{const _d=new Date(_x);return _d.getFullYear()===2001&&_d.getDate()===15}))}
function TTD_binLabel(G,ne){const c=G.columns[0],xs=G.rows.map(r=>r[c]).filter(v=>typeof v=="number"),d=xs.slice(1).map((v,i)=>v-xs[i]).filter(v=>v>0),w=d.length?Math.min(...d):1,f=v=>Number(v.toFixed(3)).toString(),x=ne[c];return typeof x=="number"?\`\${f(x-w/2)} – \${f(x+w/2)}\`:"—"}
var ttdInput={padding:"5px 8px",fontSize:12,borderRadius:6,border:"1px solid var(--border-2)",background:"var(--bg-3)",color:"var(--text-1)",outline:"none"};
function TTDAggMonth(p){return(0,h.jsxs)("div",{style:{display:"flex",flexDirection:"column",gap:10},children:[
 ttdNote(["Donne une ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"année type"}),", de janvier à décembre. Chaque jour est d'abord calculé avec l'opération choisie, puis on fait la moyenne des jours de chaque mois, toutes années confondues."]),
 ttdNote(["Pour le flux de sève : ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"Somme"}),", pas minimum ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"48"}),", facteur ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"0,5"})," → cumul journalier moyen de chaque mois (l dm⁻² j⁻¹). Pour la météo (VPD, Rn…) : Moyenne."]),
 (0,h.jsxs)("div",{style:{display:"flex",gap:6,flexWrap:"wrap"},children:[ttdSmallBtn("Réglage flux : cumul journalier",()=>{p.setOp("Somme"),p.setMin(48),p.setScale(.5)}),ttdSmallBtn("Réglage météo : moyenne",()=>{p.setOp("Moyenne"),p.setMin(0),p.setScale(1)})]})]})}
function TTDAggBins(p){const noEnv=p.envCols.length===0,cols=noEnv?p.dataCols:p.envCols,toggle=m=>p.setMonths(p.months.includes(m)?p.months.filter(x=>x!==m):[...p.months,m].sort((a,b)=>a-b));return(0,h.jsxs)("div",{style:{display:"flex",flexDirection:"column",gap:12},children:[
 ttdNote(["Regroupe les valeurs selon les ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"classes d'une autre variable"})," (VPD, Rn…) au lieu du temps : on obtient la ",(0,h.jsx)("strong",{style:{color:"var(--text-2)"},children:"courbe de réponse"})," du flux à cette variable. Ouvre ensuite le résultat dans Visualisation (graphique « Courbe XY »)."]),
 noEnv&&(0,h.jsxs)("div",{style:{fontSize:11,color:"var(--warning)",background:"var(--bg-2)",border:"1px solid var(--border-2)",borderRadius:6,padding:"8px 10px",lineHeight:1.5},children:["Aucune donnée météo chargée : la variable doit alors être une colonne des données choisies. Pour utiliser le VPD, le Rn… charge d'abord le fichier météo. ",ttdSmallBtn("Aller à Données env.",()=>p.goTab("envdata"))]}),
 (0,h.jsxs)("div",{children:[ttdLabel("Variable qui définit les classes"),(0,h.jsxs)("select",{value:p.bv,onChange:e=>p.setBv(e.target.value),style:{...ttdInput,width:"100%"},children:[(0,h.jsx)("option",{value:"",children:"— choisir une variable —"}),...cols.map(c=>(0,h.jsx)("option",{value:c,children:c},c))]})]}),
 (0,h.jsxs)("div",{style:{display:"flex",gap:16,flexWrap:"wrap",alignItems:"flex-end"},children:[
  (0,h.jsxs)("div",{children:[ttdLabel("Largeur d'une classe"),(0,h.jsx)("input",{type:"number",min:0,step:"any",value:p.bw,onChange:e=>p.setBw(Number(e.target.value)||0),style:{...ttdInput,width:90}}),(0,h.jsx)("span",{style:{fontSize:10,color:"var(--text-5)",marginLeft:6},children:"dans l'unité de la variable (ex. 5 hPa)"})]})]}),
 (0,h.jsxs)("div",{children:[ttdLabel("Heures de la journée"),(0,h.jsxs)("div",{style:{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap",fontSize:12,color:"var(--text-3)"},children:[
  ttdChip(!p.hOn,"Toute la journée",()=>p.setHOn(!1),"all"),ttdChip(p.hOn,"Seulement",()=>p.setHOn(!0),"win"),
  p.hOn&&(0,h.jsxs)(h.Fragment,{children:["de",(0,h.jsx)("input",{type:"number",min:0,max:23,value:p.h1,onChange:e=>p.setH1(Math.min(23,Math.max(0,parseInt(e.target.value)||0))),style:{...ttdInput,width:56}}),"h à",(0,h.jsx)("input",{type:"number",min:0,max:24,value:p.h2,onChange:e=>p.setH2(Math.min(24,Math.max(0,parseInt(e.target.value)||0))),style:{...ttdInput,width:56}}),"h"]})]})]}),
 (0,h.jsxs)("div",{children:[ttdLabel("Mois pris en compte (aucun coché = toute l'année)"),(0,h.jsx)("div",{style:{display:"grid",gridTemplateColumns:"repeat(12, 1fr)",gap:4},children:${JSON.stringify(MOIS)}.map((l,i)=>ttdChip(p.months.includes(i+1),l,()=>toggle(i+1),i))}),
  (0,h.jsxs)("div",{style:{display:"flex",gap:6,marginTop:6,flexWrap:"wrap"},children:[ttdSmallBtn("Toute l'année",()=>p.setMonths([])),ttdSmallBtn("Saison sèche (nov.–mai)",()=>p.setMonths([1,2,3,4,5,11,12])),ttdSmallBtn("Saison des pluies (juin–oct.)",()=>p.setMonths([6,7,8,9,10]))]})]}),
 ttdNote("Le « pas minimum par intervalle » plus bas s'applique ici à chaque classe : une classe avec moins de valeurs est ignorée (conseil : 20 ou plus).")]})}
`;
rep("function Xme(){", helpers + "function Xme(){");

// état du composant Agrégation
rep("[SCAL,SETSCAL]=(0,j.useState)(1),",
  "[SCAL,SETSCAL]=(0,j.useState)(1),[BV,SETBV]=(0,j.useState)(\"\"),[BW,SETBW]=(0,j.useState)(5),[BHON,SETBHON]=(0,j.useState)(!1),[BH1,SETBH1]=(0,j.useState)(10),[BH2,SETBH2]=(0,j.useState)(16),[BM,SETBM]=(0,j.useState)([]),SETTAB=Gt(_w=>_w.setCurrentTab),ENVC=(a.find(_s=>_s.key===\"env\")?.columns??[]).filter(_c=>!/^(TIMESTAMP|DATE)$/i.test(_c)),");

// appel backend
rep('profile:v==="profile"?"HourOfDay":void 0,profileBy:v==="profile"?m:void 0',
  'profile:v==="profile"?"HourOfDay":v==="month"?"MonthOfYear":v==="bins"?"Bins":void 0,profileBy:v==="profile"?m:v==="bins"?BV:void 0,binWidth:v==="bins"?BW:void 0,hourFrom:v==="bins"&&BHON?BH1:void 0,hourTo:v==="bins"&&BHON?BH2:void 0,months:v==="bins"&&BM.length?BM:void 0');
rep("Ie=o&&de.length>0&&!W", 'Ie=o&&de.length>0&&!W&&(v!=="bins"||!!BV)');

// rechargement d'une agrégation sauvegardée
rep('Ne.profile?(g("profile"),x(Ne.profile_by??"Mensuel")):(g("period"),f(Ne.period))',
  'Ne.profile==="MonthOfYear"?g("month"):Ne.profile==="Bins"?(g("bins"),SETBV(Ne.profile_by??"")):Ne.profile?(g("profile"),x(Ne.profile_by??"Mensuel")):(g("period"),f(Ne.period))');

// sélecteur de mode + panneaux
rep('{id:"period",label:"Période"},{id:"profile",label:"Journée moyenne"}',
  '{id:"period",label:"Période"},{id:"profile",label:"Journée moyenne"},{id:"month",label:"Mois moyen"},{id:"bins",label:"Par classes"}');
rep('v==="profile"?(0,h.jsxs)("div",{style:{display:"flex",flexDirection:"column",gap:10}',
  'v==="month"?TTDAggMonth({setOp:S,setMin:SETMINC,setScale:SETSCAL}):v==="bins"?TTDAggBins({envCols:ENVC,dataCols:pe,bv:BV,setBv:SETBV,bw:BW,setBw:SETBW,hOn:BHON,setHOn:SETBHON,h1:BH1,setH1:SETBH1,h2:BH2,setH2:SETBH2,months:BM,setMonths:SETBM,goTab:SETTAB}):v==="profile"?(0,h.jsxs)("div",{style:{display:"flex",flexDirection:"column",gap:10}');

// résultats : titre, colonne période, bouton « Ouvrir dans Visualisation »
rep('G.profile?`Journée moyenne / ${Kme(G.profile_by??"Global")}`:G.period',
  'G.profile==="MonthOfYear"?"Mois moyen (janv. → déc.)":G.profile==="Bins"?`Par classes de ${G.profile_by}`:G.profile?`Journée moyenne / ${Kme(G.profile_by??"Global")}`:G.period');
rep('children:G.profile?"Heure":t("agg.results.period")',
  'children:G.profile==="MonthOfYear"?"Mois":G.profile==="Bins"?"Classe":G.profile?"Heure":t("agg.results.period")');
rep('children:G.profile?`${String(ne.period).slice(11,13)}h`:ne.period',
  'children:G.profile==="MonthOfYear"?TTD_MOIS[parseInt(String(ne.period).slice(5,7))-1]:G.profile==="Bins"?TTD_binLabel(G,ne):G.profile?`${String(ne.period).slice(11,13)}h`:ne.period');
rep('children:t("agg.results.saved",{name:G.saved.name})})]})]})',
  'children:t("agg.results.saved",{name:G.saved.name})})]}),G.saved&&(0,h.jsx)("button",{onClick:()=>{window.__TTD_VIZ_OPEN={source:`agg_${G.saved.id}`,columns:G.columns,profile:G.profile},SETTAB("visualisation")},title:"Remplace le graphique de Visualisation par ce résultat",style:{marginLeft:12,padding:"4px 12px",fontSize:11,fontWeight:600,borderRadius:6,border:"none",background:"var(--accent)",color:"#fff",cursor:"pointer"},children:"📈 Ouvrir dans Visualisation"}),G.saved&&(0,h.jsx)("button",{onClick:()=>{window.__TTD_VIZ_OPEN={source:`agg_${G.saved.id}`,columns:G.columns,profile:G.profile,append:!0},SETTAB("visualisation")},title:"Ajoute ce résultat aux courbes déjà affichées (ex. flux + météo pour les corrélations)",style:{marginLeft:8,padding:"4px 12px",fontSize:11,fontWeight:600,borderRadius:6,border:"1px solid var(--accent-tint-30)",background:"var(--accent-tint-10)",color:"var(--accent-hover)",cursor:"pointer"},children:"➕ Ajouter au graphique"})]})');

rep('t("agg.log.success",{period:c,', 't("agg.log.success",{period:v==="month"?"Mois moyen":v==="bins"?"Par classes":v==="profile"?"Journée moyenne":c,');
rep('onClick:()=>g(ne.id)', 'onClick:()=>{g(ne.id),ne.id==="bins"&&v!=="bins"&&(S("Moyenne"),SETMINC(20),SETSCAL(1))}');

// ---------------------------------------------------------------- Visualisation
const vizHelpers = `
function ttdNorm(p){let a=1/0,b=-1/0;for(const[,v]of p)v!=null&&isFinite(v)&&(v<a&&(a=v),v>b&&(b=v));const d=b-a;return d>0?p.map(([x,v])=>[x,v==null?null:(v-a)/d]):p}
function ttdEmpty(n,txt){return{backgroundColor:"transparent",title:{text:txt,left:"center",top:"middle",textStyle:{color:n.textColor,fontSize:12}}}}
function ttdColumns(e,r){const out=[];for(const s of e){const pairs=r(s);s.columns.forEach((c,i)=>out.push({name:c,source:s.source,color:gO(s.color,i,s.columns.length),pairs:pairs[i]??[]}))}return out}
function TTDXYLine(t){const{series:e,extractPairs:r,theme:n,sourceLabel:a}=t;if(e.length<2)return ttdEmpty(n,"Ajoute 2 séries : la 1ʳᵉ pour l'axe X (ex. centre de classe), les suivantes pour les courbes.");const X=new Map;for(const[x,v]of r(e[0])[0]??[])v!=null&&isFinite(v)&&X.set(x,v);const ys=ttdColumns(e.slice(1),r),series=ys.map(y=>{const d=[];for(const[x,v]of y.pairs){const xv=X.get(x);v!=null&&isFinite(v)&&xv!==void 0&&d.push([xv,v])}return d.sort((p,q)=>p[0]-q[0]),{name:\`\${y.name} · \${a(y.source)}\`,type:"line",data:d,showSymbol:!0,symbolSize:6,lineStyle:{width:2,color:y.color},itemStyle:{color:y.color}}}).filter(s=>s.data.length>0);if(series.length===0)return ttdEmpty(n,"Aucun point commun entre la série X et les autres séries.");return{backgroundColor:"transparent",animation:!1,grid:{top:40,bottom:56,left:60,right:24,containLabel:!0},xAxis:{type:"value",scale:!0,name:e[0].columns[0],nameLocation:"middle",nameGap:30,nameTextStyle:{color:n.textColor,fontSize:11},axisLabel:{color:n.textColor,fontSize:10},splitLine:{lineStyle:{color:n.borderColor,type:"dashed"}}},yAxis:{type:"value",scale:!0,axisLabel:{color:n.textColor,fontSize:10},splitLine:{lineStyle:{color:n.borderColor,type:"dashed"}}},legend:{type:"scroll",top:4,textStyle:{color:n.textColor,fontSize:10}},tooltip:{trigger:"axis",backgroundColor:n.tooltipBg,borderColor:n.borderColor,textStyle:{color:n.tooltipText,fontSize:11}},series}}
function ttdRanks(v){const idx=v.map((x,i)=>i).sort((p,q)=>v[p]-v[q]),r=new Array(v.length);for(let i=0;i<idx.length;){let j=i;for(;j+1<idx.length&&v[idx[j+1]]===v[idx[i]];)j++;const m=(i+j)/2+1;for(let k=i;k<=j;k++)r[idx[k]]=m;i=j+1}return r}
function ttdSpearman(a,b){const x=[],y=[];for(const[k,v]of a){const w=b.get(k);w!==void 0&&(x.push(v),y.push(w))}if(x.length<10)return{rho:null,n:x.length};const rx=ttdRanks(x),ry=ttdRanks(y),n=x.length,mx=rx.reduce((s,v)=>s+v,0)/n,my=ry.reduce((s,v)=>s+v,0)/n;let sxy=0,sxx=0,syy=0;for(let i=0;i<n;i++){const dx=rx[i]-mx,dy=ry[i]-my;sxy+=dx*dy,sxx+=dx*dx,syy+=dy*dy}return{rho:sxx>0&&syy>0?sxy/Math.sqrt(sxx*syy):null,n}}
function TTDCorr(t){const{series:e,extractPairs:r,theme:n,sourceLabel:a}=t,cols=ttdColumns(e,r).map(c=>{const m=new Map;for(const[x,v]of c.pairs)v!=null&&isFinite(v)&&m.set(x,v);return{...c,map:m}}).filter(c=>c.map.size>0);if(cols.length<2)return ttdEmpty(n,"Ajoute au moins 2 colonnes (ex. flux journalier et VPD journalier).");const names=cols.map(c=>c.name),data=[],info={};for(let i=0;i<cols.length;i++)for(let j=0;j<cols.length;j++){const{rho,n:k}=i===j?{rho:1,n:cols[i].map.size}:ttdSpearman(cols[i].map,cols[j].map);info[i+","+j]=k,data.push([j,i,rho==null?"-":Math.round(rho*100)/100])}const fs=cols.length>14?8:10;return{backgroundColor:"transparent",animation:!1,grid:{top:16,bottom:110,left:10,right:70,containLabel:!0},xAxis:{type:"category",data:names,axisLabel:{color:n.textColor,fontSize:fs,rotate:40,interval:0},splitArea:{show:!1}},yAxis:{type:"category",data:names,inverse:!0,axisLabel:{color:n.textColor,fontSize:fs,interval:0}},visualMap:{min:-1,max:1,calculable:!1,orient:"vertical",right:0,top:"middle",itemHeight:180,text:["+1","−1"],textStyle:{color:n.textColor,fontSize:10},inRange:{color:["#e34948","#f0efec","#2a78d6"]}},tooltip:{backgroundColor:n.tooltipBg,borderColor:n.borderColor,textStyle:{color:n.tooltipText,fontSize:11},formatter:p=>{const[j,i,v]=p.value;return\`\${names[i]} (\${a(cols[i].source)})<br/>× \${names[j]} (\${a(cols[j].source)})<br/>ρ = <b>\${v}</b> · \${info[i+","+j]} dates communes\`}},series:[{type:"heatmap",data,label:{show:cols.length<=16,fontSize:fs,color:"#0b0b0b"},itemStyle:{borderColor:"transparent",borderWidth:1}}]}}
`;
rep("function iye(t){", vizHelpers + "function iye(t){");
rep('[O,W]=(0,j.useState)(!1),V=(0,j.useRef)(O);V.current=O;',
  '[O,W]=(0,j.useState)(!1),V=(0,j.useRef)(O);V.current=O;const[NRM,SETNRM]=(0,j.useState)(!1);(0,j.useEffect)(()=>{const _p=window.__TTD_VIZ_OPEN;if(a!=="visualisation"||!_p)return;window.__TTD_VIZ_OPEN=null;const _mk=(src,cols,k)=>({id:`${Date.now()}-${k}`,source:src,columns:cols,color:ug[k%ug.length],visible:!0,yAxis:0});if(_p.append){const _cols=_p.columns.filter(c=>c!=="Nombre de points");z(_prev=>[..._prev,_mk(_p.source,_cols,_prev.length)])}else if(_p.profile==="Bins"){const _c=_p.columns.find(c=>c.endsWith("(centre de classe)")),_y=_p.columns.filter(c=>c!==_c&&c!=="Nombre de points");z([_mk(_p.source,[_c],0),_mk(_p.source,_y,1)]),E("xyline")}else z([_mk(_p.source,_p.columns,0)]),E("line")},[a]);');
rep('{id:"scatter",labelKey:"viz.chartType.scatter"}',
  '{id:"scatter",labelKey:"viz.chartType.scatter"},{id:"xyline",labelKey:"viz.chartType.xyline"},{id:"corr",labelKey:"viz.chartType.corr"}');
rep('if(I==="scatter")return iye(',
  'if(I==="xyline")return TTDXYLine({series:xt,extractPairs:st,theme:{textColor:ge,borderColor:oe,tooltipBg:le,tooltipText:Ce,accent:Xe},sourceLabel:Dr,t});if(I==="corr")return TTDCorr({series:xt,extractPairs:st,theme:{textColor:ge,borderColor:oe,tooltipBg:le,tooltipText:Ce,accent:Xe},sourceLabel:Dr,t});if(I==="scatter")return iye(');
rep('I==="scatter"&&(0,h.jsx)("p",{style:{fontSize:10,color:"var(--text-5)",margin:"8px 0 0",lineHeight:1.5},children:t("viz.chartType.scatterHint")}),',
  'I==="scatter"&&(0,h.jsx)("p",{style:{fontSize:10,color:"var(--text-5)",margin:"8px 0 0",lineHeight:1.5},children:t("viz.chartType.scatterHint")}),I==="xyline"&&(0,h.jsx)("p",{style:{fontSize:10,color:"var(--text-5)",margin:"8px 0 0",lineHeight:1.5},children:t("viz.chartType.xylineHint")}),I==="corr"&&(0,h.jsx)("p",{style:{fontSize:10,color:"var(--text-5)",margin:"8px 0 0",lineHeight:1.5},children:t("viz.chartType.corrHint")}),');
rep("const zt=st(tt),Ot=I===\"combo\"", "const zt=NRM?st(tt).map(ttdNorm):st(tt),Ot=I===\"combo\"");
rep("},[xt,vt,T,n,Dr,I,O,q,Z,J,t])", "},[xt,vt,T,n,Dr,I,O,q,Z,J,t,NRM])");
// étiquettes de mois pour le « Mois moyen » (année fictive 2001, jour 15)
rep('series:St};},[xt,vt,T,n,Dr,I,O,q,Z,J,t,NRM])', 'series:St}},[xt,vt,T,n,Dr,I,O,q,Z,J,t,NRM])', 0);
rep('xAxis:{type:"time",axisLabel:{color:ge,fontSize:10,hideOverlap:!0}',
  'xAxis:{type:"time",...(TTD_isMonth(St)?{min:new Date(2001,0,1).getTime(),max:new Date(2001,11,31).getTime()}:{}),axisLabel:{color:ge,fontSize:10,hideOverlap:!0,...(TTD_isMonth(St)?{formatter:_v=>TTD_MOIS[new Date(_v).getMonth()]}:{})}');
// case à cocher « Indexer de 0 à 1 »
rep('children:t("viz.invertYHint")})',
  'children:t("viz.invertYHint")}),(0,h.jsxs)("button",{onClick:()=>SETNRM(!NRM),style:{marginTop:8,width:"100%",display:"flex",alignItems:"center",gap:10,padding:"8px 12px",borderRadius:8,background:NRM?"var(--accent-tint-10)":"var(--bg-2)",border:`1px solid ${NRM?"var(--accent-tint-30)":"var(--border-2)"}`,cursor:"pointer",transition:"all 0.15s"},children:[(0,h.jsx)("div",{style:{width:16,height:16,borderRadius:4,background:NRM?"var(--accent)":"var(--bg-3)",border:NRM?"none":"1px solid var(--border-3)",display:"flex",alignItems:"center",justifyContent:"center"},children:NRM&&(0,h.jsx)("span",{style:{color:"#fff",fontSize:11,fontWeight:700},children:"✓"})}),(0,h.jsx)("span",{style:{fontSize:12,color:NRM?"var(--accent-hover)":"var(--text-3)"},children:t("viz.normalize")})]}),NRM&&(0,h.jsx)("p",{style:{fontSize:10,color:"var(--text-5)",margin:"8px 0 0",lineHeight:1.5},children:t("viz.normalizeHint")})');


// ---------------------------------------------------------------- Entraînement IA : préréglage « Longs trous »
rep('precis:{epochs:140,patience:22,n_layers:3,d_model:256,d_ffn:512,MIT_weight:2.5,max_nan_frac:.75}}',
  'precis:{epochs:140,patience:22,n_layers:3,d_model:256,d_ffn:512,MIT_weight:2.5,max_nan_frac:.75},longs:{epochs:140,patience:15,n_layers:3,d_model:256,d_ffn:512,MIT_weight:2.5,max_nan_frac:.5}}');
rep('eO=[{id:"rapide",label:"Rapide"},{id:"standard",label:"Standard"},{id:"precis",label:"Précis"}]',
  'eO=[{id:"rapide",label:"Rapide"},{id:"standard",label:"Standard"},{id:"precis",label:"Précis"},{id:"longs",label:"Longs trous (≤ 7 j)"}]');
rep('Ge.model_config=J6[v],Ge.time_features=[...b?["doy_sin","doy_cos"]:[],...w?["hod_sin","hod_cos"]:[],...C?["tabs"]:[]];',
  'Ge.model_config=J6[v],Ge.time_features=[...b?["doy_sin","doy_cos"]:[],...w?["hod_sin","hod_cos"]:[],...C?["tabs"]:[]];v==="longs"&&(Ge.window=144,Ge.stride=36,Ge.step="30min",Ge.block_mask_prob=.5,Ge.time_features=["hod_sin","hod_cos","doy_sin","doy_cos","tabs"]);');
rep('return`Réglage « ${eO.find(De=>De.id===v)?.label} » : ${_e.epochs} époques',
  'return v==="longs"?`Réglage « Longs trous (≤ 7 j) » : ${_e.epochs} époques, ${_e.n_layers} couches, d_model ${_e.d_model}, poids imputation ×${_e.MIT_weight}, fenêtre de 3 jours et apprentissage « sonde masquée » (une sonde entière cachée dans une fenêtre sur deux). Calendrier activé automatiquement ; ajoute la météo (VPD, Rn, LAI). Testé sur Niakhar 2 : erreur sur le flux journalier divisée par 2 environ par rapport à l\'interpolation pour des trous de 7 jours. Combine-le avec « taille max de trou = 7 j » dans Gap Filling.`:`Réglage « ${eO.find(De=>De.id===v)?.label} » : ${_e.epochs} époques');

// ---------------------------------------------------------------- Couleurs distinctes par courbe
// Plusieurs colonnes d'une même série recevaient des nuances d'une seule teinte, et
// deux séries pouvaient se partager la même couleur : illisible dès 4-5 courbes.
// Chaque courbe prend désormais sa propre couleur de la palette, dans l'ordre ; une
// série à une seule colonne garde la couleur choisie par l'utilisateur.
rep('return zt.map((rr,Tt)=>{const nr=gO(tt.color,Tt,tt.columns.length),',
  'const _off=xt.slice(0,xt.indexOf(tt)).reduce((_a,_s)=>_a+_s.columns.length,0);return zt.map((rr,Tt)=>{const nr=xt.length===1&&tt.columns.length===1?tt.color:ug[(_off+Tt)%ug.length],');

fs.writeFileSync(outPath, s);
console.log("patch appliqué :", outPath);
