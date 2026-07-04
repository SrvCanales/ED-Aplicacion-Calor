import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS AVANZADOS
# ==========================================
st.set_page_config(page_title="Ruta de Aprendizaje: Calor 1D", layout="wide", initial_sidebar_state="expanded")

# Inyección de CSS para armonizar la interfaz y crear componentes limpios
st.markdown("""
    <style>
    .main-card {
        background-color: #f8fafc;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .system-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #1565C0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .alert-card {
        background-color: #fff7ed;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ffedd5;
        color: #9a3412;
        font-weight: 500;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Paleta de Colores (Códigos HEX)
C_STAT = "#1565C0"  # Azul: Estacionaria
C_TRANS = "#C62828" # Rojo: Transitoria
C_GEN = "#6A1B9A"   # Morado: Solución General
C_SPACE = "#2E7D32" # Verde: Espacial
C_TIME = "#EF6C00"  # Naranjo: Temporal (a_n(t))

# ==========================================
# GESTIÓN DEL ESTADO (Ruta Progresiva)
# ==========================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'math_data' not in st.session_state:
    st.session_state.math_data = {}

def avanzar():
    st.session_state.step += 1

# ==========================================
# BARRA LATERAL (Progreso del Estudiante)
# ==========================================
st.sidebar.markdown(f"### 🗺️ Tu Progreso")
pasos = [
    "1. Planteando la EDP",
    "2. Homogeneizando fronteras",
    "3. Valores y funciones propias",
    "4. Expansión por autofunciones",
    "5. Coeficientes temporales",
    "6. Solución General y Simulador"
]

for i, paso in enumerate(pasos):
    if i + 1 < st.session_state.step:
        st.sidebar.markdown(f"✅ <span style='color:gray;'>{paso}</span>", unsafe_allow_html=True)
    elif i + 1 == st.session_state.step:
        st.sidebar.markdown(f"👉 **<span style='color:{C_GEN};'>{paso}</span>**", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"🔒 <span style='color:lightgray;'>{paso}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Código de Color:**")
st.sidebar.markdown(f"■ <span style='color:{C_STAT}; font-weight:bold;'>Estacionaria</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TRANS}; font-weight:bold;'>Transitoria</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_GEN}; font-weight:bold;'>General</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_SPACE}; font-weight:bold;'>Espacial</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TIME}; font-weight:bold;'>Temporal</span>", unsafe_allow_html=True)

# ==========================================
# MOTOR MATEMÁTICO CENTRAL
# ==========================================
x, t, tau = sp.symbols('x t tau')
n_sym = sp.Symbol('n', integer=True, positive=True)

def calcular_matematicas(L_str, alpha_str, F_str, A_str, B_str, f_str):
    """Calcula algebraicamente la solución general utilizando n simbólico."""
    L = sp.sympify(L_str, locals={'pi': sp.pi})
    alpha = sp.sympify(alpha_str)
    F = sp.sympify(F_str, locals={'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    A = sp.sympify(A_str)
    B = sp.sympify(B_str)
    f = sp.sympify(f_str, locals={'x': x, 'pi': sp.pi, 'sin': sp.sin})
    
    # 2. Homogeneización
    w = sp.simplify(A + (x / L) * (B - A))
    F_tilde = sp.simplify(F - sp.diff(w, t) + alpha**2 * sp.diff(w, x, 2))
    f_tilde = sp.simplify(f - w.subs(t, 0))
    
    # 3. Componente Espacial
    lam_n = n_sym * sp.pi / L
    phi_n = sp.sin(lam_n * x)
    
    # 5. Componente Temporal Proyectada (a_n(t))
    q_n_expr = (2/L) * sp.integrate(F_tilde * phi_n, (x, 0, L))
    
    # Para la simulación numérica posterior, fijamos internamente N=6 términos representativos
    v_sol_num = 0
    for n_val in range(1, 7):
        lam_val = n_val * sp.pi / L
        phi_val = sp.sin(lam_val * x)
        q_n_val = (2/L) * sp.integrate(F_tilde * phi_val, (x, 0, L))
        a_n_0 = (2/L) * sp.integrate(f_tilde * phi_val, (x, 0, L))
        
        factor_k = alpha**2 * lam_val**2
        int_part = sp.integrate(q_n_val.subs(t, tau) * sp.exp(factor_k * tau), tau)
        int_eval = int_part.subs(tau, t) - int_part.subs(tau, 0)
        
        a_n_t = sp.exp(-factor_k * t) * (int_eval + a_n_0)
        v_sol_num += a_n_t * phi_val

    u_final_num = sp.simplify(w + v_sol_num)
    u_num_func = sp.lambdify((x, t), u_final_num, modules=['numpy', 'math'])
    
    st.session_state.math_data = {
        'L': L, 'alpha': alpha, 'F': F, 'A': A, 'B': B, 'f': f,
        'w': w, 'F_tilde': F_tilde, 'f_tilde': f_tilde,
        'lam_n': lam_n, 'phi_n': phi_n, 'q_n_expr': q_n_expr,
        'u_num_func': u_num_func, 'L_num': float(L.evalf())
    }

# ==========================================
# DIÁLOGO DE AYUDA DE ALTO IMPACTO VISUAL
# ==========================================
@st.dialog("📖 Profundización Matemática: Homogeneización", width="large")
def mostrar_ayuda_profunda(w_d, F_t_d, f_t_d):
    st.markdown("## 1.2 Homogeneización de las condiciones de frontera")
    st.markdown("""
    El método de separación de variables funciona de manera natural cuando las **condiciones de frontera son homogéneas**, es decir,
    $$v(0,t)=v(L,t)=0$$
    Como nuestro problema original no necesariamente cumple esta condición, transformaremos la temperatura total en la suma de dos contribuciones:
    * **Una parte estacionaria (o pseudo-estacionaria)** $w(x,t)$, encargada de satisfacer las condiciones de frontera.
    * **Una parte transitoria** $v(x,t)$, que contendrá toda la evolución temporal restante y tendrá fronteras homogéneas.
    
    En otras palabras:
    """)
    st.latex(r"u(x,t)=w(x,t)+v(x,t)")
    
    st.markdown("### 🛠️ Construcción de la función auxiliar")
    st.markdown("""
    Existen **infinitas funciones** que satisfacen las mismas condiciones de frontera. Sin embargo, elegimos la interpolación lineal entre ambos extremos porque es la opción más simple y, además, cumple que
    $$\\frac{\\partial^2 w}{\\partial x^2}=0$$
    por lo que **no introduce términos adicionales** en la ecuación diferencial. Gracias a ello, la EDP transformada conserva prácticamente la misma estructura que la original.
    """)
    
    st.markdown("**Tu función auxiliar calculada en tiempo real:**")
    st.latex(f"w(x,t) = {sp.latex(w_d)}")
    
    st.markdown("""> **Observación importante:**
    > Podríamos escoger otra función que también cumpliera las condiciones de frontera (por ejemplo, agregando un seno o un polinomio que valga cero en ambos extremos). Sin embargo, esa elección produciría un término extra $\\alpha^2 w_{xx}-w_t$, complicando innecesariamente la ecuación para $v(x,t)$. La solución física final **no cambia**, pero la separación entre la parte estacionaria y la transitoria deja de ser tan clara.""")
    
    st.markdown("### 🔄 Problema transformado")
    st.markdown("Al sustituir $u(x,t)=w(x,t)+v(x,t)$ en la ecuación de calor, obtenemos un nuevo problema para $v(x,t)$, cuya fuente y condición inicial se transforman en:")
    st.latex(f"\\tilde{{F}}(x,t) = {sp.latex(F_t_d)}")
    st.latex(f"\\tilde{{f}}(x) = v(x,0) = {sp.latex(f_t_d)}")
    
    st.markdown("""En adelante resolveremos este nuevo problema, que posee **fronteras homogéneas** y es adecuado para aplicar el método de separación de variables.
    
    > **💡 Idea clave:**
    > La homogeneización **no modifica la solución física del problema**. Únicamente cambia la forma de representarla. Elegir una función $w$ apropiada hace que $v$ describa únicamente la parte transitoria del fenómeno, simplificando considerablemente el análisis matemático.""")

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🔥 Simulador de EDP: Calor 1D Análitico")
st.markdown("Sigue esta ruta guiada interactiva para resolver tu Problema de Valor Inicial y de Frontera (PVIF) paso a paso mediante expresiones algebraicas generales.")
st.markdown("---")

# ---------------------------------------------------------
# ETAPA 1: PLANTEAMIENTO (COMPLETAMENTE REDISEÑADA)
# ---------------------------------------------------------
if st.session_state.step >= 1:
    st.header("1. Planteando la EDP")
    
    # Grid armonizado para el ingreso de datos
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            in_L = st.text_input("Longitud de la barra (L):", value="pi", disabled=(st.session_state.step > 1))
            in_alpha = st.text_input("Difusividad térmica (α):", value="1", disabled=(st.session_state.step > 1))
            in_F = st.text_input("Fuente externa de calor F(x,t):", value="exp(-t)*sin(2*x)", disabled=(st.session_state.step > 1))
        with col2:
            in_A = st.text_input("Condición de frontera Izquierda u(0,t):", value="10", disabled=(st.session_state.step > 1))
            in_B = st.text_input("Condición de frontera Derecha u(L,t):", value="50", disabled=(st.session_state.step > 1))
            in_f = st.text_input("Perfil de temperatura Inicial u(x,0):", value="3*sin(4*pi*x)", disabled=(st.session_state.step > 1))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- DISPLAY EN TIEMPO REAL CON FORMATO DE LLAVE ---
    st.markdown("### 📋 Visualización del Sistema en Tiempo Real:")
    try:
        L_s = sp.sympify(in_L, locals={'pi': sp.pi})
        alpha_s = sp.sympify(in_alpha)
        F_s = sp.sympify(in_F, locals={'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
        A_s = sp.sympify(in_A)
        B_s = sp.sympify(in_B)
        f_s = sp.sympify(in_f, locals={'x': x, 'pi': sp.pi, 'sin': sp.sin})
        
        # Cálculos dinámicos inmediatos para alimentar la ayuda oculta de forma responsiva
        w_dyn = sp.simplify(A_s + (x / L_s) * (B_s - A_s))
        F_t_dyn = sp.simplify(F_s - sp.diff(w_dyn, t) + alpha_s**2 * sp.diff(w_dyn, x, 2))
        f_t_dyn = sp.simplify(f_s - w_dyn.subs(t, 0))
        
        latex_sistema = rf"""
        \begin{cases}
        \frac{{\partial u}}{{\partial t}} = {sp.latex(alpha_s^2)} \frac{{\partial^2 u}}{{\partial x^2}} + {sp.latex(F_s)}, & 0 < x < {sp.latex(L_s)}, \; t > 0 \\
        u(0,t) = {sp.latex(A_s)}, & t > 0 \\
        u({sp.latex(L_s)},t) = {sp.latex(B_s)}, & t > 0 \\
        u(x,0) = {sp.latex(f_s)}, & 0 \le x \le {sp.latex(L_s)}
        \end{cases}
        """
    except Exception:
        # Fallback amigable mientras el usuario se encuentra digitando expresiones incompletas
        latex_sistema = r"\begin{cases} \text{Esperando expresiones válidas...} \end{cases}"
        w_dyn, F_t_dyn, f_t_dyn = "w(x,t)", r"\tilde{F}(x,t)", r"\tilde{f}(x)"

    st.markdown("<div class='system-card'>", unsafe_allow_html=True)
    st.latex(latex_sistema)
    st.markdown("</div>", unsafe_allow_html=True)

    # Botón sutil de ayuda interactiva que despliega el modal detallado
    col_help, _ = st.columns([1, 3])
    with col_help:
        if st.button("ℹ️ Ver Fundamentos Matemáticos y Homogeneización", use_container_width=True):
            mostrar_ayuda_profunda(w_dyn, F_t_dyn, f_t_dyn)

    # Recordatorio explícito solicitado para guiar la proactividad del estudiante
    st.markdown("""
        <div class='alert-card'>
        💡 <b>Reto de Aprendizaje:</b> ¡Intenta homogeneizar las fronteras antes de ejecutar la próxima celda!, luego compara tu procedimiento y resultado.
        </div>
    """, unsafe_allow_html=True)
        
    if st.session_state.step == 1:
        if st.button("Guardar Parámetros y Avanzar 🚀", type="primary"):
            with st.spinner("Procesando núcleo matemático analítico..."):
                calcular_matematicas(in_L, in_alpha, in_F, in_A, in_B, in_f)
            avanzar()
            st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 2: HOMOGENEIZACIÓN
# ---------------------------------------------------------
if st.session_state.step >= 2:
    data = st.session_state.math_data
    st.header("2. Homogeneizando las fronteras")
    
    st.markdown(f"Proponemos que la solución se divide en una parte estacionaria y una transitoria:")
    st.latex(fr"\color{{{C_GEN}}}{{u(x,t)}} = \color{{{C_STAT}}}{{w(x,t)}} + \color{{{C_TRANS}}}{{v(x,t)}}")
    
    st.markdown("La parte estacionaria calculada es:")
    st.latex(fr"\color{{{C_STAT}}}{{w(x,t) = {sp.latex(data['w'])}}}")
    
    st.markdown("Al sustituir esto en la EDP original, nuestro nuevo problema transitorio tiene una fuente y condición inicial modificadas:")
    st.latex(fr"\color{{{C_TRANS}}}{{\tilde{{F}}(x,t)}} = {sp.latex(data['F_tilde'])}")
    st.latex(fr"\color{{{C_TRANS}}}{{\tilde{{f}}(x)}} = {sp.latex(data['f_tilde'])}")
        
    if st.session_state.step == 2:
        if st.button("Buscar Valores Propios 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 3: VALORES Y FUNCIONES PROPIAS
# ---------------------------------------------------------
if st.session_state.step >= 3:
    data = st.session_state.math_data
    st.header("3. Valores y Funciones Propias Espaciales")
    
    st.markdown("Separando variables en el lado homogéneo, el problema espacial arroja:")
    st.latex(fr"\color{{{C_SPACE}}}{{\lambda_n^2}} = \left( {sp.latex(data['lam_n'])} \right)^2")
    st.latex(fr"\color{{{C_SPACE}}}{{\phi_n(x)}} = {sp.latex(data['phi_n'])}")

    if st.session_state.step == 3:
        if st.button("Plantear Expansión 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 4: EXPANSIÓN Y STURM-LIOUVILLE
# ---------------------------------------------------------
if st.session_state.step >= 4:
    data = st.session_state.math_data
    st.header("4. Sturm-Liouville y Expansión")
    
    st.markdown("Con nuestra base espacial lista, expandimos nuestras incógnitas temporales transformadas en coeficientes $a_n(t)$:")
    st.latex(fr"\color{{{C_TRANS}}}{{v(x,t)}} = \sum_{{n=1}}^{{\infty}} \color{{{C_TIME}}}{{a_n(t)}} \color{{{C_SPACE}}}{{\sin\left(\frac{{n\pi x}}{{{sp.latex(data['L'])}}}\right)}}")
    st.latex(fr"\color{{{C_TRANS}}}{{\tilde{{F}}(x,t)}} = \sum_{{n=1}}^{{\infty}} \color{{{C_TIME}}}{{q_n(t)}} \color{{{C_SPACE}}}{{\sin\left(\frac{{n\pi x}}{{{sp.latex(data['L'])}}}\right)}}")

    if st.session_state.step == 4:
        if st.button("Resolver Coeficientes Temporales 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 5: COEFICIENTES TEMPORALES a_n(t)
# ---------------------------------------------------------
if st.session_state.step >= 5:
    data = st.session_state.math_data
    st.header("5. Obteniendo los Coeficientes Temporales")
    
    st.markdown("Al introducir las series en la EDP, proyectamos todo por ortogonalidad y transformamos la ecuación parcial en infinitas Ecuaciones Diferenciales Ordinarias (EDOs) para los coeficientes algebraicos de tiempo $a_n(t)$:")
    st.latex(fr"\color{{{C_TIME}}}{{a_n'(t)}} + {sp.latex(data['alpha']**2)} \color{{{C_SPACE}}}{{\left(\frac{{n\pi}}{{{sp.latex(data['L'])}}}\right)^2}} \color{{{C_TIME}}}{{a_n(t)}} = \color{{{C_TIME}}}{{q_n(t)}}")
    
    st.markdown("Donde los coeficientes de la fuente proyectada se calculan formalmente como:")
    st.latex(fr"\color{{{C_TIME}}}{{q_n(t)}} = {sp.latex(data['q_n_expr'])}")

    if st.session_state.step == 5:
        if st.button("Ensamblar Solución Final y Simular 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 6: SOLUCIÓN Y SIMULADOR
# ---------------------------------------------------------
if st.session_state.step >= 6:
    data = st.session_state.math_data
    st.header("6. Solución General y Simulador Físico")
    
    st.markdown("**Modo de Visualización de la Solución:**")
    vista = st.radio("Selecciona qué componente de la solución deseas analizar:", 
                     ["Solución General u(x,t)", "Sólo Estacionaria w(x,t)", "Sólo Transitoria v(x,t)"],
                     horizontal=True)
    
    L_tex = sp.latex(data['L'])
    w_tex = sp.latex(data['w'])
    sum_tex = r"\sum_{n=1}^{\infty} \color{" + C_TIME + r"}{a_n(t)} \color{" + C_SPACE + r"}{\sin\left(\frac{n\pi x}{" + L_tex + r"}\right)}"
    
    st.markdown("<div style='background-color:#ffffff; padding:20px; border-radius:10px; border:1px solid #e5e7eb;'>", unsafe_allow_html=True)
    if vista == "Sólo Estacionaria w(x,t)":
        st.latex(fr"\color{{{C_STAT}}}{{w(x,t)}} = \color{{{C_STAT}}}{{{w_tex}}}")
    elif vista == "Sólo Transitoria v(x,t)":
        st.latex(fr"\color{{{C_TRANS}}}{{v(x,t)}} = {sum_tex}")
    else:
        u_inf_tex = (fr"\color{{{C_STAT}}}{{{w_tex}}} + " if data['w'] != 0 else "") + sum_tex
        st.latex(fr"\color{{{C_GEN}}}{{u(x,t)}} = {u_inf_tex}")
    st.markdown("</div>", unsafe_allow_html=True)

    # SIMULADOR NUMÉRICO ASOCIADO
    st.subheader("Simulación Térmica Interactiva")
    t_val = st.slider(f"Flujo del Tiempo (t)", 0.0, 5.0, 0.0, 0.05)
    
    x_vals = np.linspace(0, data['L_num'], 250)
    u_vals = data['u_num_func'](x_vals, t_val)
    if np.isscalar(u_vals):
        u_vals = np.ones_like(x_vals) * u_vals

    fig, ax = plt.subplots(figsize=(10, 2), dpi=120)
    extent = [0, data['L_num'], 0, 1]
    
    im = ax.imshow(np.array([u_vals, u_vals]), cmap='inferno', aspect='auto', extent=extent, vmin=np.min(u_vals)-0.1, vmax=np.max(u_vals)+1)
    ax.set_yticks([])
    ax.set_xlabel('Posición de la barra $x$', fontweight='bold', color='#475569')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.3, pad=0.5)
    cbar.set_label(f'Temperatura a los {t_val:.2f}s', fontweight='bold', color='#475569')
    cbar.outline.set_visible(False)
    
    st.pyplot(fig)
