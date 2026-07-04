import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y COLORES
# ==========================================
st.set_page_config(page_title="Ruta de Aprendizaje: Calor 1D", layout="wide", initial_sidebar_state="expanded")

# Paleta de Colores (Códigos HEX)
C_STAT = "#1565C0"  # Azul: Estacionaria
C_TRANS = "#C62828" # Rojo: Transitoria
C_GEN = "#6A1B9A"   # Morado: Solución General
C_SPACE = "#2E7D32" # Verde: Espacial
C_TIME = "#EF6C00"  # Naranjo: Temporal

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

def calcular_matematicas(L_str, alpha_str, F_str, A_str, B_str, f_str, N):
    """Calcula y almacena toda la matemática pesada en el estado una sola vez."""
    L = sp.sympify(L_str, locals={'pi': sp.pi})
    alpha = sp.sympify(alpha_str)
    F = sp.sympify(F_str, locals={'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    A = sp.sympify(A_str)
    B = sp.sympify(B_str)
    f = sp.sympify(f_str, locals={'x': x, 'pi': sp.pi, 'sin': sp.sin})
    
    # 2. Homogenización
    w = sp.simplify(A + (x / L) * (B - A))
    F_tilde = sp.simplify(F - sp.diff(w, t) + alpha**2 * sp.diff(w, x, 2))
    f_tilde = sp.simplify(f - w.subs(t, 0))
    
    # 3. Espacial
    lam_n = n_sym * sp.pi / L
    phi_n = sp.sin(lam_n * x)
    
    # 5. Temporal & Serie
    v_sol = 0
    q_n_expr = (2/L) * sp.integrate(F_tilde * phi_n, (x, 0, L)) # Expresión formal
    
    for n in range(1, N + 1):
        lam_val = n * sp.pi / L
        phi_val = sp.sin(lam_val * x)
        
        q_n_val = (2/L) * sp.integrate(F_tilde * phi_val, (x, 0, L))
        T_n_0 = (2/L) * sp.integrate(f_tilde * phi_val, (x, 0, L))
        
        factor_k = alpha**2 * lam_val**2
        int_part = sp.integrate(q_n_val.subs(t, tau) * sp.exp(factor_k * tau), tau)
        int_eval = int_part.subs(tau, t) - int_part.subs(tau, 0)
        
        T_n = sp.exp(-factor_k * t) * (int_eval + T_n_0)
        v_sol += T_n * phi_val

    u_final_num = sp.simplify(w + v_sol)
    u_num_func = sp.lambdify((x, t), u_final_num, modules=['numpy', 'math'])
    
    st.session_state.math_data = {
        'L': L, 'alpha': alpha, 'F': F, 'A': A, 'B': B, 'f': f, 'N': N,
        'w': w, 'F_tilde': F_tilde, 'f_tilde': f_tilde,
        'lam_n': lam_n, 'phi_n': phi_n, 'q_n_expr': q_n_expr,
        'v_sol': v_sol, 'u_num_func': u_num_func, 'L_num': float(L.evalf())
    }


# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🔥 Simulador de EDP: Calor 1D")
st.markdown("Sigue esta ruta guiada para resolver tu Problema de Valor Inicial y de Frontera (PVIF) paso a paso.")
st.markdown("---")

# ---------------------------------------------------------
# ETAPA 1: PLANTEAMIENTO
# ---------------------------------------------------------
if st.session_state.step >= 1:
    st.header("1. Planteando la EDP")
    
    col1, col2 = st.columns(2)
    with col1:
        in_L = st.text_input("Longitud L:", value="pi", disabled=(st.session_state.step > 1))
        in_alpha = st.text_input("Difusividad α:", value="1", disabled=(st.session_state.step > 1))
        in_F = st.text_input("Fuente F(x,t):", value="exp(-t)*sin(2*x)", disabled=(st.session_state.step > 1))
        in_N = st.slider("Términos (Serie):", 1, 15, 3, disabled=(st.session_state.step > 1))
    with col2:
        in_A = st.text_input("Frontera u(0,t):", value="10", disabled=(st.session_state.step > 1))
        in_B = st.text_input("Frontera u(L,t):", value="50", disabled=(st.session_state.step > 1))
        in_f = st.text_input("Inicial u(x,0):", value="3*sin(4*pi*x)", disabled=(st.session_state.step > 1))
    
    with st.expander("📖 ¿Por qué necesitamos tantas condiciones iniciales y de frontera?"):
        st.markdown("""
        Una EDP de calor tiene una derivada temporal (primer orden) y dos derivadas espaciales (segundo orden). 
        Para que el problema tenga una **solución única**, necesitamos "anclar" las constantes de integración de esas derivadas:
        * **1 Condición Inicial:** Describe la "foto" de la temperatura en el instante $t=0$.
        * **2 Condiciones de Frontera:** Describen qué le ocurre a los extremos de la barra ($x=0$ y $x=L$) en todo instante de tiempo.
        """)
        
    if st.session_state.step == 1:
        if st.button("Guardar Parámetros y Avanzar 🚀", type="primary"):
            with st.spinner("Calculando modelo matemático..."):
                calcular_matematicas(in_L, in_alpha, in_F, in_A, in_B, in_f, in_N)
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

    with st.expander("📖 ¿Por qué homogenizamos las condiciones de frontera?", expanded=False):
        st.markdown(r"""
        <p>Es completamente normal preguntarse por qué realizamos esta transformación. El método de separación de variables necesita trabajar con condiciones de frontera homogéneas, por lo que reescribimos la temperatura como</p>
        <center>
        $u(x,t)= \color{#1565C0}{w(x,t)} + \color{#C62828}{v(x,t)}$
        </center>
        <ul>
        <li><span style="color:#1565C0;"><b>Parte estacionaria:</b></span> absorbe las fronteras fijas.</li>
        <li><span style="color:#C62828;"><b>Parte transitoria:</b></span> describe únicamente la evolución hacia ese equilibrio.</li>
        </ul>
        <hr>
        <b>💡 Idea clave</b><br>
        La homogeneización <b>no cambia la solución física del problema</b>. Únicamente cambia la forma de escribirla. Elegimos una función lineal porque $w_{xx}=0$, evitando que aparezcan términos basura innecesarios en nuestra nueva EDP.
        """, unsafe_allow_html=True)
        
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

    with st.expander("📖 ¿Qué significan físicamente estos valores?"):
        st.markdown(f"""
        Piensa en la barra de calor como si fuera la cuerda de una guitarra. 
        Al estar sujeta en los extremos (fronteras homogéneas 0), la cuerda sólo puede vibrar en ciertos patrones perfectos u "ondas estacionarias".
        
        * Las **<span style='color:{C_SPACE};'>funciones propias $\phi_n(x)$</span>** son esos patrones perfectos (los armónicos).
        * Los **<span style='color:{C_SPACE};'>valores propios $\lambda_n$</span>** están relacionados con la rapidez espacial con la que decae la temperatura de cada patrón.
        """, unsafe_allow_html=True)

    if st.session_state.step == 3:
        if st.button("Plantear Expansión 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 4: EXPANSIÓN
# ---------------------------------------------------------
if st.session_state.step >= 4:
    data = st.session_state.math_data
    st.header("4. Sturm-Liouville y Expansión")
    
    st.markdown("Con nuestra base espacial lista, expandimos nuestras incógnitas temporales y la fuente modificada:")
    st.latex(fr"\color{{{C_TRANS}}}{{v(x,t)}} = \sum_{{n=1}}^{{\infty}} \color{{{C_TIME}}}{{T_n(t)}} \color{{{C_SPACE}}}{{\sin\left(\frac{{n\pi x}}{{{sp.latex(data['L'])}}}\right)}}")
    st.latex(fr"\color{{{C_TRANS}}}{{\tilde{{F}}(x,t)}} = \sum_{{n=1}}^{{\infty}} \color{{{C_TIME}}}{{q_n(t)}} \color{{{C_SPACE}}}{{\sin\left(\frac{{n\pi x}}{{{sp.latex(data['L'])}}}\right)}}")

    with st.expander("📖 El Principio de Superposición (La magia de Fourier)"):
        st.markdown(r"""
        Joseph Fourier descubrió que **cualquier** distribución inicial de calor (incluso si es muy extraña o tiene picos abruptos) se puede construir sumando infinitas ondas suaves de senos y cosenos.
        
        Al proponer esta sumatoria, estamos diciéndole a la matemática: *"No resuelvas todo el caos de una vez. Dime cómo se comporta cada armónico $n$ individualmente, y yo luego los sumaré todos"*.""")

    if st.session_state.step == 4:
        if st.button("Resolver Coeficientes Temporales 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 5: COEFICIENTES TEMPORALES
# ---------------------------------------------------------
if st.session_state.step >= 5:
    data = st.session_state.math_data
    st.header("5. Obteniendo los Coeficientes Temporales")
    
    st.markdown("Al introducir las series en la EDP, proyectamos todo por ortogonalidad y transformamos la ecuación parcial en infinitas Ecuaciones Diferenciales Ordinarias (EDOs):")
    st.latex(fr"\color{{{C_TIME}}}{{T_n'(t)}} + {sp.latex(data['alpha']**2)} \color{{{C_SPACE}}}{{\left(\frac{{n\pi}}{{{sp.latex(data['L'])}}}\right)^2}} \color{{{C_TIME}}}{{T_n(t)}} = \color{{{C_TIME}}}{{q_n(t)}}")
    
    st.markdown("Donde los coeficientes de la fuente proyectada se calculan como:")
    st.latex(fr"\color{{{C_TIME}}}{{q_n(t)}} = {sp.latex(data['q_n_expr'])}")

    with st.expander("📖 De EDP a EDO: ¿Qué acaba de pasar?"):
        st.markdown(f"""
        Este es el corazón del método. Integrar variables dependientes de $x$ y $t$ al mismo tiempo es casi imposible.
        Pero al usar la propiedad de **ortogonalidad** de los senos (donde multiplicar dos senos diferentes e integrarlos da 0), logramos "matar" la dimensión espacial.
        
        Nos quedamos únicamente con EDOs que dependen del **<span style='color:{C_TIME};'>tiempo $t$</span>**, las cuales se resuelven con métodos clásicos de Cálculo 2 (factor integrante).
        """, unsafe_allow_html=True)

    if st.session_state.step == 5:
        if st.button("Ensamblar Solución Final y Simular 🚀"): avanzar(); st.rerun()
    st.markdown("---")

# ---------------------------------------------------------
# ETAPA 6: SOLUCIÓN Y SIMULADOR
# ---------------------------------------------------------
if st.session_state.step >= 6:
    data = st.session_state.math_data
    st.header("6. Solución General y Simulador Físico")
    
    # Toggle de visualización (Requisito Especial)
    st.markdown("**Modo de Visualización de la Solución:**")
    vista = st.radio("Selecciona qué componente de la solución deseas analizar:", 
                     ["Solución General u(x,t)", "Sólo Estacionaria w(x,t)", "Sólo Transitoria v(x,t)"],
                     horizontal=True)
    
    L_tex = sp.latex(data['L'])
    w_tex = sp.latex(data['w'])
    sum_tex = r"\sum_{n=1}^{\infty} \color{" + C_TIME + r"}{T_n(t)} \color{" + C_SPACE + r"}{\sin\left(\frac{n\pi x}{" + L_tex + r"}\right)}"
    
    st.markdown("<div style='background-color:#ffffff; padding:20px; border-radius:10px; border:1px solid #e5e7eb;'>", unsafe_allow_html=True)
    if vista == "Sólo Estacionaria w(x,t)":
        st.latex(fr"\color{{{C_STAT}}}{{w(x,t)}} = \color{{{C_STAT}}}{{{w_tex}}}")
    elif vista == "Sólo Transitoria v(x,t)":
        st.latex(fr"\color{{{C_TRANS}}}{{v(x,t)}} = {sum_tex}")
    else:
        u_inf_tex = (fr"\color{{{C_STAT}}}{{{w_tex}}} + " if data['w'] != 0 else "") + sum_tex
        st.latex(fr"\color{{{C_GEN}}}{{u(x,t)}} = {u_inf_tex}")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📖 La Física detrás de la Fórmula"):
        st.markdown(f"""
        Al ensamblar todo, vemos la belleza de la termodinámica:
        * **El largo plazo (La suma tiende a 0):** A medida que el tiempo $t$ avanza hacia el infinito, el factor $e^{{-kt}}$ dentro de los $\color{{{C_TIME}}}{{T_n(t)}}$ hace que toda la parte <span style='color:{C_TRANS}; font-weight:bold;'>Transitoria</span> desaparezca.
        * **El Equilibrio:** Lo único que sobrevive en la barra de metal al cabo de muchas horas es la parte <span style='color:{C_STAT}; font-weight:bold;'>Estacionaria $w(x,t)$</span>.
        """, unsafe_allow_html=True)

    # SIMULADOR NUMÉRICO
    st.subheader("Simulación Térmica Interactiva")
    t_val = st.slider(f"Flujo del Tiempo (t)", 0.0, 5.0, 0.0, 0.05)
    
    # Dibujar Matplotlib
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
