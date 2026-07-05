import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(page_title="Ruta de Aprendizaje: Calor 1D", layout="wide", initial_sidebar_state="expanded")

# Paleta de Colores Suaves (Material Design)
C_STAT = "#1976D2"  # Azul (Estacionaria)
C_TRANS = "#D32F2F" # Rojo (Transitoria)
C_GEN = "#7B1FA2"   # Morado (Solución General)
C_SPACE = "#388E3C" # Verde (Espacial)
C_TIME = "#F57C00"  # Naranjo (Temporal)

def c_sym(symbol, color):
    """Función auxiliar para colorear SOLO el símbolo matemático, sin afectar paréntesis ni otros elementos."""
    return rf"\color{{{color}}}{{{symbol}}}"

# ==========================================
# GESTIÓN DEL ESTADO
# ==========================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'math_data' not in st.session_state:
    st.session_state.math_data = {}

def avanzar():
    st.session_state.step += 1

# ==========================================
# BARRA LATERAL (Progreso)
# ==========================================
st.sidebar.markdown("### 🗺️ Tu Progreso")
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
st.sidebar.markdown("**Código de Color (Variables):**")
st.sidebar.markdown(f"■ <span style='color:{C_STAT}; font-weight:bold;'>Estacionaria ($w$)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TRANS}; font-weight:bold;'>Transitoria ($v$)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_GEN}; font-weight:bold;'>General ($u$)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_SPACE}; font-weight:bold;'>Espacial ($\phi$)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TIME}; font-weight:bold;'>Temporal ($a_n$)</span>", unsafe_allow_html=True)

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
    
    # Homogeneización
    w = sp.simplify(A + (x / L) * (B - A))
    F_tilde = sp.simplify(F - sp.diff(w, t) + alpha**2 * sp.diff(w, x, 2))
    f_tilde = sp.simplify(f - w.subs(t, 0))
    
    # Componente Espacial
    lam_n = n_sym * sp.pi / L
    phi_n = sp.sin(lam_n * x)
    
    # Componente Temporal Proyectada
    q_n_expr = (2/L) * sp.integrate(F_tilde * phi_n, (x, 0, L))
    
    # Simulación numérica (N=6 términos)
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
# DIÁLOGO DE AYUDA (Conceptos Separados)
# ==========================================
@st.dialog("📖 Profundización Matemática: Homogeneización", width="large")
def mostrar_ayuda_profunda(w_d, F_t_d, f_t_d):
    st.subheader("El Problema de las Fronteras No Homogéneas")
    st.markdown("El método de separación de variables requiere **condiciones de frontera nulas** ($0$). Cuando tenemos temperaturas fijas distintas de cero en los extremos, debemos transformar el problema.")
    
    st.subheader("La Sustitución")
    st.markdown("Separamos la temperatura total en dos contribuciones físicas distintas:")
    u_xt = f"{c_sym('u', C_GEN)}(x,t)"
    w_xt = f"{c_sym('w', C_STAT)}(x,t)"
    v_xt = f"{c_sym('v', C_TRANS)}(x,t)"
    st.latex(f"{u_xt} = {w_xt} + {v_xt}")
    
    st.markdown("1. **Perfil Estacionario** ($w$): Una función lineal que conecta las temperaturas de los extremos. Al ser una recta, su segunda derivada espacial es cero, por lo que no altera la estructura de la EDP.")
    st.latex(f"{w_xt} = {sp.latex(w_d)}")
    
    st.markdown("2. **Perfil Transitorio** ($v$): Es la diferencia entre la temperatura real y el perfil estacionario. Por definición, vale cero en los extremos.")
    
    st.subheader("El Problema Homogéneo Resultante")
    st.markdown("Al sustituir la solución propuesta en la EDP original, obtenemos una nueva ecuación para la parte transitoria, donde la fuente y la condición inicial absorben los efectos de las fronteras:")
    F_tilde_xt = f"{c_sym(r'\tilde{F}', C_TRANS)}(x,t)"
    f_tilde_x = f"{c_sym(r'\tilde{f}', C_TRANS)}(x)"
    st.latex(f"{F_tilde_xt} = {sp.latex(F_t_d)}")
    st.latex(f"{f_tilde_x} = {c_sym('v', C_TRANS)}(x,0) = {sp.latex(f_t_d)}")
    
    st.info("💡 **Interpretación Física:** La homogeneización no cambia la física del problema; simplemente desplaza nuestro sistema de referencia. Ahora modelamos cómo la temperatura *evoluciona* hacia el estado estacionario, en lugar de modelar la temperatura absoluta directamente.")

# ==========================================
# INTERFAZ PRINCIPAL - ETAPA 1
# ==========================================
st.title("🔥 Simulador de EDP: Calor 1D Analítico")
st.markdown("Sigue esta ruta guiada interactiva para resolver tu Problema de Valor Inicial y de Frontera (PVIF) paso a paso.")
st.divider()

st.header("1. Planteando la EDP")
st.markdown("Define los parámetros físicos y geométricos de tu sistema unidimensional.")

# Contenedor nativo con borde para los inputs
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        in_L = st.text_input("Longitud de la barra (L):", value="L")
        in_alpha = st.text_input("Difusividad térmica (α):", value="alpha")
        in_F = st.text_input("Fuente externa de calor F(x,t):", value="Q_0 * exp(-t)")
    with col2:
        in_A = st.text_input("Frontera Izquierda u(0,t):", value="T_1")
        in_B = st.text_input("Frontera Derecha u(L,t):", value="T_2")
        in_f = st.text_input("Perfil Inicial u(x,0):", value="T_0 * sin(pi*x/L)")

transformaciones = (standard_transformations + (implicit_multiplication_application,))

def parsear_seguro(expr_str):
    if not expr_str.strip():
        raise ValueError("Expresión vacía")
    expr = parse_expr(expr_str, transformations=transformaciones)
    reemplazos = {sim: (x if sim.name == 'x' else (t if sim.name == 't' else sim)) for sim in expr.free_symbols}
    return expr.subs(reemplazos)

st.subheader("Modelo Matemático (Tiempo Real)")

try:
    L_s = parsear_seguro(in_L)
    alpha_s = parsear_seguro(in_alpha)
    F_s = parsear_seguro(in_F)
    A_s = parsear_seguro(in_A)
    B_s = parsear_seguro(in_B)
    f_s = parsear_seguro(in_f)
    
    w_dyn = sp.simplify(A_s + (x / L_s) * (B_s - A_s))
    F_t_dyn = sp.simplify(F_s - sp.diff(w_dyn, t) + alpha_s**2 * sp.diff(w_dyn, x, 2))
    f_t_dyn = sp.simplify(f_s - w_dyn.subs(t, 0))
    
    # Colorear solo la función incógnita
    u_sym = c_sym('u', C_GEN)
    alpha_term = sp.latex(alpha_s**2) if not alpha_s.is_number else sp.latex(alpha_s**2)
    
    latex_sistema = rf"""
    \begin{{cases}}
    \frac{{\partial {u_sym}}}{{\partial t}} = {alpha_term} \frac{{\partial^2 {u_sym}}}{{\partial x^2}} + {sp.latex(F_s)}, & 0 < x < {sp.latex(L_s)}, \quad t > 0 \\[8pt]
    {u_sym}(0,t) = {sp.latex(A_s)}, & t > 0 \\[8pt]
    {u_sym}({sp.latex(L_s)},t) = {sp.latex(B_s)}, & t > 0 \\[8pt]
    {u_sym}(x,0) = {sp.latex(f_s)}, & 0 \le x \le {sp.latex(L_s)}
    \end{{cases}}
    """
    
    # Contenedor nativo para el sistema de ecuaciones
    with st.container(border=True):
        st.latex(latex_sistema)
        
    col_help, _ = st.columns([1, 2])
    with col_help:
        if st.button("ℹ️ Ver Fundamentos de Homogeneización", use_container_width=True):
            mostrar_ayuda_profunda(w_dyn, F_t_dyn, f_t_dyn)

except Exception:
    with st.container(border=True):
        st.latex(r"\text{Esperando expresiones matemáticas válidas...}")

st.info("💡 **Reto de Aprendizaje:** Identifica mentalmente cuáles términos causan que el problema no sea homogéneo antes de avanzar.")

if st.button("Guardar Parámetros y Avanzar 🚀", type="primary"):
    # Mocking el proceso matemático para el flujo de la UI
    calcular_matematicas(in_L, in_alpha, in_F, in_A, in_B, in_f)
    if st.session_state.step == 1:
        avanzar()
    st.rerun()

st.divider()

# ---------------------------------------------------------
# ETAPA 2: HOMOGENEIZACIÓN
# ---------------------------------------------------------
if st.session_state.step >= 2:
    data = st.session_state.math_data
    st.header("2. Homogeneizando las fronteras")
    
    st.markdown("Proponemos separar la solución en dos contribuciones físicas:")
    u_xt = f"{c_sym('u', C_GEN)}(x,t)"
    w_xt = f"{c_sym('w', C_STAT)}(x,t)"
    v_xt = f"{c_sym('v', C_TRANS)}(x,t)"
    st.latex(f"{u_xt} = {w_xt} + {v_xt}")
    
    st.markdown("Interpolando linealmente las fronteras, la **parte estacionaria** resulta en:")
    st.latex(f"{w_xt} = {c_sym(sp.latex(data['w']), C_STAT)}")
    
    st.markdown("Sustituyendo en la EDP, nuestro **nuevo problema transitorio** queda sujeto a:")
    F_tilde_xt = f"{c_sym(r'\tilde{F}', C_TRANS)}(x,t)"
    f_tilde_x = f"{c_sym(r'\tilde{f}', C_TRANS)}(x)"
    
    with st.container(border=True):
        st.latex(f"{F_tilde_xt} = {sp.latex(data['F_tilde'])}")
        st.latex(f"{f_tilde_x} = {sp.latex(data['f_tilde'])}")
        
    if st.session_state.step == 2:
        if st.button("Buscar Valores Propios 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 3: VALORES Y FUNCIONES PROPIAS
# ---------------------------------------------------------
if st.session_state.step >= 3:
    data = st.session_state.math_data
    st.header("3. Valores y Funciones Propias Espaciales")
    
    st.markdown("Resolviendo el problema de Sturm-Liouville asociado a las fronteras homogéneas nulas en ambos extremos, obtenemos la base espacial:")
    
    lam_n = f"{c_sym(r'\lambda_n', C_SPACE)}"
    phi_n = f"{c_sym(r'\phi_n', C_SPACE)}(x)"
    
    with st.container(border=True):
        st.latex(f"{lam_n} = {sp.latex(data['lam_n'])}")
        st.latex(f"{phi_n} = {sp.latex(data['phi_n'])}")

    if st.session_state.step == 3:
        if st.button("Plantear Expansión 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 4: EXPANSIÓN Y STURM-LIOUVILLE
# ---------------------------------------------------------
if st.session_state.step >= 4:
    data = st.session_state.math_data
    st.header("4. Sturm-Liouville y Expansión")
    
    st.markdown("Expresamos la solución transitoria y la fuente modificada como combinaciones lineales (series de Fourier) de nuestras funciones propias espaciales:")
    
    v_xt = f"{c_sym('v', C_TRANS)}(x,t)"
    F_tilde_xt = f"{c_sym(r'\tilde{F}', C_TRANS)}(x,t)"
    a_n = c_sym('a_n', C_TIME)
    q_n = c_sym('q_n', C_TIME)
    sin_term = f"{c_sym(r'\sin', C_SPACE)}\left(\\frac{{n\pi x}}{{{sp.latex(data['L'])}}}\\right)"
    
    with st.container(border=True):
        st.latex(f"{v_xt} = \sum_{{n=1}}^{{\infty}} {a_n}(t) \cdot {sin_term}")
        st.latex(f"{F_tilde_xt} = \sum_{{n=1}}^{{\infty}} {q_n}(t) \cdot {sin_term}")

    if st.session_state.step == 4:
        if st.button("Resolver Coeficientes Temporales 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 5: COEFICIENTES TEMPORALES a_n(t)
# ---------------------------------------------------------
if st.session_state.step >= 5:
    data = st.session_state.math_data
    st.header("5. Obteniendo los Coeficientes Temporales")
    
    st.markdown("Proyectando la ecuación original sobre la base ortogonal, la EDP se desacopla en infinitas Ecuaciones Diferenciales Ordinarias (EDOs) para el tiempo:")
    
    a_n_prime = f"{c_sym(r'a_n^\prime', C_TIME)}(t)"
    a_n = f"{c_sym('a_n', C_TIME)}(t)"
    q_n = f"{c_sym('q_n', C_TIME)}(t)"
    factor = f"{sp.latex(data['alpha']**2)} \left(\\frac{{n\pi}}{{{sp.latex(data['L'])}}}\\right)^2"
    
    with st.container(border=True):
        st.latex(f"{a_n_prime} + {factor} {a_n} = {q_n}")
        
    st.markdown("Donde los coeficientes de la fuente proyectada calculados por el producto interno son:")
    st.latex(f"{q_n} = {sp.latex(data['q_n_expr'])}")

    if st.session_state.step == 5:
        if st.button("Ensamblar Solución Final y Simular 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 6: SOLUCIÓN Y SIMULADOR
# ---------------------------------------------------------
if st.session_state.step >= 6:
    data = st.session_state.math_data
    st.header("6. Solución General y Simulador Físico")
    
    vista = st.radio("Filtro de Visualización Matemática:", 
                     ["Solución General completa", "Solo Estacionaria", "Solo Transitoria"],
                     horizontal=True)
    
    L_tex = sp.latex(data['L'])
    w_tex = sp.latex(data['w'])
    
    # Construcción limpia usando la función c_sym
    a_n_term = f"{c_sym('a_n', C_TIME)}(t)"
    sin_term = f"{c_sym(r'\sin', C_SPACE)}\left(\\frac{{n\pi x}}{{{L_tex}}}\\right)"
    sum_tex = rf"\sum_{{n=1}}^{{\infty}} {a_n_term} \cdot {sin_term}"
    
    with st.container(border=True):
        if vista == "Solo Estacionaria":
            st.latex(f"{c_sym('w', C_STAT)}(x,t) = {c_sym(w_tex, C_STAT)}")
        elif vista == "Solo Transitoria":
            st.latex(f"{c_sym('v', C_TRANS)}(x,t) = {sum_tex}")
        else:
            w_part = f"{c_sym(w_tex, C_STAT)} + " if data['w'] != 0 else ""
            st.latex(f"{c_sym('u', C_GEN)}(x,t) = {w_part} {sum_tex}")

    # SIMULADOR NUMÉRICO
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
