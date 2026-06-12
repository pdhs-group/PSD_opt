# Granulation WMCPBE und Validierungsskripte Gebrauchsanleitung

## Installation

```python
git clone <repo-url>
cd PSD_opt
pip install -e ./pbe-core
pip install -e ./mcpbe
```

## 1. Zusammensetzung des WMCPBE-Solvers

`wmcpbe_granulation` ist eine vereinfachte Version eines gewichteten Monte-Carlo-PBE-Solvers.

Die zentrale Eingangsklasse liegt hier:

```python
wmcpbe_granulation.mcpbe.MCPBESolver
```

`MCPBESolver` kombiniert verschiedene Funktionsmodule durch Mehrfachvererbung:

```python
class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    pass
```

Aufgaben der Module:

| Datei | Hauptaufgabe |
| --- | --- |
| `mcpbe.py` | Definiert die finale Eingangsklasse `MCPBESolver`, kombiniert alle mixins. |
| `mcpbe_base.py` | Grundrahmen: Konstruktor, Eingang für Partikelinitialisierung, Sampler-Initialisierung, Kapazitätserweiterung, Haupt-Zeitfortschrittsschleife, `solve_repeats()`, Spalten hinzufügen/löschen. |
| `mcpbe_initialization.py` | Logik für Anfangspartikel, inklusive explizite Initialisierung aus `V_flat + W_init` und einfache Modellinitialisierung. |
| `mcpbe_agg.py` | Agglomeration-Logik: Agglomerationskern, Agglomerationsneigung, Kollisionspaar wählen, ein Agglomerationsereignis ausführen, Gewichte und Sampler aktualisieren. |
| `mcpbe_break.py` | Breakage-Logik: Bruchrate, Fragmentverteilungs-CDF, ein Bruchereignis ausführen, Gewichte und Sampler aktualisieren. |
| `mcpbe_time_helper.py` | Hilfsfunktionen für Zeitschritt und Gewicht-packet, inklusive Zeitschrittberechnung für Agglomeration/Breakage/Mix-Prozess. |
| `reconstruction_mixin.py` | Rekonstruktionslogik, zum Komprimieren/Resampling der Partikeldarstellung, wenn zu viele Rechenpartikel existieren. |
| `mcpbe_post.py` | Nachbearbeitung: Momente und PSD-CDF nach Zeit-Snapshots berechnen. |
| `fenwick.py` | Fenwick-tree Sampler, für schnelles zufälliges Sampling nach Ereignisneigungen. |

## 2. Grundlaufreihenfolge: Beispiel `solve_repeats()`

### 2.1 Warum `solve_repeats()`

`solve_repeats()` ist momentan die einfachste und stabilste externe Schnittstelle für WMCPBE. Sie erledigt automatisch:

1. Zufallsseed für jede Wiederholungsrechnung erzeugen.
2. Aktuelles solver-template tief kopieren.
3. Für jede Kopie unabhängigen Zufallszahlengenerator setzen.
4. Jede solver-Kopie mit derselben Gruppe Anfangspartikel und Gewichte initialisieren.
5. Internes `solve()` aufrufen und eine Monte-Carlo-Trajektorie rechnen.
6. Nachbearbeitungsmethode aufrufen und zeitabhängige Momente berechnen.
7. Ergebnisse aller Wiederholungsläufe zusammenfassen.

Monte-Carlo-Methode ist im Kern eine zufällige Methode. Ein einzelner Lauf ist nur eine zufällige Trajektorie und kann deutliche Zufallsschwankung haben. Zweck von Wiederholungsrechnungen:

- Zufallsfehler reduzieren;
- Varianz oder Standardabweichung der Ergebnisse schätzen;
- statistische Zuverlässigkeit erhöhen;
- beim Vergleich verschiedener Parametergruppen vermeiden, eine einzelne Trajektorie als Modelltrend zu interpretieren.

Deshalb normalerweise `solve_repeats(N=...)` benutzen, nicht nur einmal `solve()` aufrufen.

### 2.2 Empfohlener minimaler Aufruf

Bei externer Nutzung normalerweise zuerst ein solver-template ohne automatische Initialisierung erstellen:

```python
from wmcpbe_granulation import MCPBESolver

solver = MCPBESolver(
    dim=2,
    t_vec=t_vec,
    verbose=False,
    load_attr=False,
    init=False,
)
```

Dann Parameter explizit setzen, zum Beispiel:

```python
solver.process_type = "breakage"      # "agglomeration" | "breakage" | "mix"
solver.COLEVAL = 3                    # Typ des Agglomerationskerns
solver.CORR_BETA = 1e-3               # Koeffizient des Agglomerationskerns
solver.BREAKRVAL = 1                  # Modell für Bruchrate
solver.BREAKFVAL = 2                  # Modell für Fragmentverteilung
solver.pl_P1 = 3e-2
solver.pl_P2 = 1.0
solver.pl_P3 = 3e-2
solver.pl_P4 = 1.0
solver.pl_v = 1.0
solver.pl_q = 1.0
solver.G = 1.0
solver.alpha_prim = np.ones(dim ** 2)
```

Am Ende mit expliziten Anfangspartikeln und Gewichten laufen lassen. `V_flat` und `W_init` müssen explizit definiert werden, später erklärt:

```python
results, psd_info = solver.solve_repeats(
    N=10,
    base_seed=42,
    maxiter=int(1e9),
    init_Vc=False,
    Vc=Vc,
    V_flat=V_flat,
    W_init=W_init,
)
```

### 2.3 Hauptparameter von `solve_repeats()`

| Parameter | Bedeutung |
| --- | --- |
| `N` | Anzahl der Wiederholungsrechnungen. Größeres `N` macht statistischen Mittelwert stabiler, aber Rechenkosten höher. |
| `base_seed` | Basis-Zufallsseed. Wenn `seeds` nicht übergeben wird, wird daraus `N` unabhängige seeds erzeugt. |
| `seeds` | Optional explizite seed-Sequenz. Länge muss gleich `N` sein. |
| `maxiter` | Maximale Ereigniszahl pro Wiederholungsrechnung, gegen unendlich lange Rechnung. |
| `init_Vc` | Ob Kontrollvolumen intern nach `a0/n0` initialisiert wird. Bei explizitem `Vc` auf `False` setzen. |
| `Vc` | Kontrollvolumen. Bei externer Initialisierung mit `V_flat + W_init` normalerweise nötig. |
| `V_flat` | Anfangs-Partikelvolumenarray, Form `(dim + 1, N_particles)`. Erste `dim` Zeilen sind Komponentenvolumina, letzte Zeile ist Gesamtpartikelvolumen. |
| `W_init` | Gewicht jedes Rechenpartikels, zeigt wie viele reale Partikel dieser Rechenpartikel vertritt. Länge muss gleich Spaltenzahl von `V_flat` sein. |
| `psd_enable` | Ob PSD-Nachbearbeitung berechnet wird. Default `False`. |
| `psd_basis` | Gewichtsbasis der PSD, normalerweise `"volume"` oder `"number"`. |
| `psd_x_grid` | Wenn gegeben, Ausgabe von `Q(x)` auf festem Partikelgrößengitter. |
| `psd_Q_grid` | Wenn gegeben, Ausgabe von `x(Q)` auf festen Quantilen. |

Rückgabewert:

```python
results, psd_info
```

Dabei ist `results` eine Liste, jedes Element entspricht einem Wiederholungslauf:

```python
{
    "seed_info": ...,
    "t_vec": ...,
    "moments": ...,
}
```

Die Form von `moments` ist normalerweise `(3, 3, T)`, also `mu[i, j, t]`. Bei 1D wird hauptsächlich `mu[i, 0, t]` benutzt.

## 3. Agglomeration, Breakage, Zeitfortschritt, Gewicht, batch-wise und Rekonstruktion

### 3.1 Grundbedeutung der Gewichte

Ein "Rechenpartikel" in WMCPBE muss nicht nur ein reales Partikel vertreten. Es kann ein Gewicht `W[k]` haben:

```python
W[k] = Anzahl realer Partikel, die vom k-ten Rechenpartikel vertreten wird
```

Damit kann der solver mit wenigen Rechenpartikeln sehr viele reale Partikel darstellen. Beim Sampling und bei der Momentberechnung werden Gewichte benutzt, zum Beispiel Moment in 1D:

```python
mu_i = sum_k W[k] * V[k]**i / Vc
```

In 2D:

```python
mu_ij = sum_k W[k] * V1[k]**i * V3[k]**j / Vc
```

### 3.2 batch-wise / packet Ereignisse

Ein normales Monte-Carlo-Ereignis kann man verstehen als: ein Ereignis behandelt ein reales Partikel oder ein Paar realer Partikel. Die gewichtete Version erlaubt für bessere Effizienz, dass ein Ereignis eine kleine Gruppe realer Ereignisse vertritt, also packet oder batch-wise Ereignis.

Zwei wichtige Parameter:

```python
agg_dW_max
break_dW_max
```

Sie kontrollieren, wie viel Gewicht ein Agglomerations-/Bruchereignis maximal verbraucht.

- `agg_dW_max = 1.0`: ein Agglomerationsereignis vertritt maximal 1 reales Agglomerations-packet.
- `break_dW_max = 1.0`: ein Bruchereignis vertritt maximal 1 reales Bruch-packet.
- Größere Werte lassen jedes Ereignis mehr reales Gewicht fortschreiten, normalerweise schneller, aber statistische/numerische Approximation gröber.

In `simple_validation.py`, wenn man nahe an normale event-by-event-Rechnung kommen will, beide setzen auf:

```python
"agg_dW_max": 1.0,
"break_dW_max": 1.0,
```

Das ist die übliche Art, batch-wise zu schließen oder zu minimieren, wie in Dokument und Beispiel gemeint.

### 3.3 Grober Ablauf von agglomeration

Agglomerationslogik liegt hauptsächlich in `mcpbe_agg.py`.

Ein Agglomerationsereignis ungefähr:

1. Nach aktuellem Partikelzustand Agglomerationsneigung `_r_agg` jedes Partikels berechnen.
2. Mit Fenwick sampler nach `_r_agg` zufällig erstes Partikel `i` wählen.
3. Nach Agglomerationskern, Gewicht und Akzeptanzwahrscheinlichkeit zweites Partikel `j` wählen.
4. Nach `agg_dW_max / agg_dW_min / agg_dW_mode` Gewichtspaketgröße `dW` dieses Ereignisses berechnen.
5. Neues Partikel erstellen, Komponentenvolumen ist `Vi + Vj`, Gewicht ist `dW`.
6. Von Elternpartikel-Gewichten `dW` abziehen; bei Selbstagglomeration vom selben Elternpartikel `2*dW` abziehen.
7. Wenn Elterngewicht auf 0 oder darunter fällt, diesen Rechenpartikel löschen.
8. Agglomerationsneigungen und Sampler neu aufbauen.
9. Wenn aktueller Prozess `mix` ist, auch Breakage-Sampler aktualisieren.

### 3.4 Grober Ablauf von breakage

Breakage-Logik liegt hauptsächlich in `mcpbe_break.py`.

Ein Bruchereignis ungefähr:

1. Bruchrate und Bruchneigung `_break_rate` jedes Partikels berechnen.
2. Nach `_break_rate` zufällig ein Mutterpartikel `k` wählen.
3. Nach `break_dW_max` und aktuellem Gewicht des Mutterpartikels packet-Größe `dW` für diesen Bruch bestimmen.
4. Nach `BREAKFVAL / pl_v / pl_q` Fragmentverteilungs-CDF konstruieren oder wiederverwenden.
5. Aus dem Volumen des Mutterpartikels mehrere Fragmente zufällig erzeugen.
6. Jedes Fragment als neuen Rechenpartikel hinzufügen, Gewicht ist `dW`.
7. Mutterpartikelgewicht um `dW` reduzieren; wenn Restgewicht 0 ist, Mutterpartikel löschen.
8. Breakage-Sampler aktualisieren; wenn aktueller Prozess `mix` ist, auch Agglomerations-Sampler aktualisieren.

### 3.5 Zeitfortschrittslogik

Zeitfortschrittslogik liegt hauptsächlich in `mcpbe_base.py` und `mcpbe_time_helper.py`.

Kernidee:

- Die Summe der Neigungen aller aktuell möglichen Ereignisse bestimmt die Zeitskala des nächsten Ereignisses.
- Agglomeration, Breakage und Mix-Prozess haben jeweils entsprechende Zeitschrittstrategie.
- `solve()` wählt intern nach `process_type`, ob Agglomeration, Breakage oder Konkurrenz von beiden ausgeführt wird.
- Nach jedem Ereignis wird aktuelle Zeit `current_time` aktualisiert, und beim Überschreiten gespeicherter Zeitpunkte wird Snapshot gespeichert.

`process_type` unterstützt:

```python
"agglomeration"
"breakage"
"mix"
```

### 3.6 Was macht reconstruction

Rekonstruktionslogik liegt in `reconstruction_mixin.py`.

Mit dem Simulationsfortschritt kann die Zahl der Rechenpartikel schnell wachsen. Zum Beispiel erzeugt Breakage ständig neue Fragmente, Agglomeration/Mix kann auch die Zustandsverteilung komplexer machen. Ziel der Rekonstruktion:

- Anzahl der Rechenpartikel kontrollieren;
- viele ähnliche Partikel zu weniger repräsentativen Partikeln komprimieren;
- wichtige Momente oder Verteilungsmerkmale möglichst erhalten;
- spätere Rechenkosten reduzieren.

Häufige Parameter:

| Parameter | Bedeutung |
| --- | --- |
| `recon_enable` | Ob Rekonstruktion aktiviert wird. |
| `recon_method` | Rekonstruktionsmethode, z.B. `"RS"`, `"2PM"`, `"4PM"`, `"4PMC"`, `"QMX"`. |
| `recon_N_max` | Wenn aktive Rechenpartikelzahl diesen Wert überschreitet, Rekonstruktion triggern. |
| `recon_bins` | Anzahl Rekonstruktionsgitter. |
| `recon_RS_target` | Zielpartikelzahl für RS-Resampling. |

Beispiel Rekonstruktion aktivieren:

```python
"recon_enable": True,
"recon_method": "4PMC",
"recon_N_max": 4000,
"recon_bins": 30,
"recon_RS_target": 1000,
```

Beispiel Rekonstruktion schließen:

```python
"recon_enable": False,
```

Hinweis: normalerweise reicht `4PMC`; diese Methode hat insgesamt höchste Genauigkeit.

## 4. Eigenen process hinzufügen: Beispiel nucleation

Wenn ein neuer Prozess hinzugefügt werden soll, z.B. `nucleation`, wird empfohlen, entlang der existierenden Agglomeration/Breakage-Struktur zu erweitern, nicht die Logik direkt in `solve()` zu stecken.

Empfohlener Implementierungspfad:

### 4.1 Neues Prozessmodul hinzufügen

Empfohlen neue ähnliche Datei:

```text
mcpbe_nucleation.py
```

Und mixin definieren:

```python
class MCPBENucleation:
    ...
```

Dieses Modul sollte enthalten:

- Vorbereitungsmethode für nucleation-Parameter;
- Berechnungsmethode für nucleation-Neigung oder Gesamtrate;
- Ausführungsmethode für einzelnes nucleation-Ereignis, z.B. `_do_one_nucleation()`;
- Wartungsmethoden für nucleation-bezogene Sampler oder Ratenarrays.

### 4.2 Vererbung der Eingangsklasse ändern

In `mcpbe.py` das neue mixin zu `MCPBESolver` hinzufügen:

```python
class MCPBESolver(MCPBEPost, MCPBENucleation, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    pass
```

Die Vererbungsreihenfolge sollte sicherstellen, dass `MCPBEBase` Methoden im neuen mixin aufrufen kann.

### 4.3 `process_type`-Unterstützung ändern

An folgenden Stellen muss `process_type`-Entscheidung erweitert werden:

- `mcpbe_base.py`
  - `_initialize_samplers()`
  - `solve()`
  - `_maybe_double_control_volume()` falls passend
  - debug-/Statuscheck-Logik falls passend
- `mcpbe_time_helper.py`
  - Zeitschrittstrategie für nucleation hinzufügen, z.B. `_build_nucleation_dt_strategy()`.

Wenn kombinierte Prozesse unterstützt werden sollen, müssen eventuell neue Modi definiert werden:

```python
"nucleation"
"nucleation_agglomeration"
"nucleation_breakage"
"mix"
```

### 4.4 Gewichtsaktualisierung

Die Kernschwierigkeit eines neuen process ist normalerweise nicht "neue Partikel erzeugen", sondern Konsistenz von Gewicht und Momenten erhalten.

Beim Hinzufügen eines neuen Ereignisses muss klar sein:

- Volumen des neuen Partikels;
- Gewicht des neuen Partikels;
- ob Gewicht alter Partikel verbraucht wird;
- ob Kontrollvolumen `Vc` geändert wird;
- welche Ratenarrays nach dem Ereignis aktualisiert werden müssen;
- welche Sampler rebuild oder lokale update brauchen;
- ob `V_save / W_save / Vc_save` Nachbearbeitungs-Snapshots beeinflusst werden.

Zum Beispiel, wenn nucleation "neue Partikel aus kontinuierlicher Phase" bedeutet, verbraucht es eventuell kein Gewicht existierender Partikel, aber fügt neue Rechenpartikel hinzu:

```python
self._append_particle_column(new_volume)
self.W[new_idx] = dW_nuc
```

Danach muss aktualisiert werden:

- Agglomerationsrate `_r_agg`, weil neue Partikel an Agglomeration teilnehmen;
- Bruchrate `_break_rate`, wenn neue Partikel auch brechen können;
- entsprechender Fenwick sampler;
- falls nötig reconstruction triggern.

## 5. Grundidee von `granulation/validation.py`

`granulation/validation.py` ist ein leichter Validierungs-Wrapper nur für `wmcpbe_granulation`.

Er wird hauptsächlich benutzt für:

- mehrere WMCPBE-Parametergruppen mit demselben Anfangspartikelzustand laufen lassen;
- Einfluss verschiedener batch-wise Parameter, Rekonstruktionsparameter, Zufallsseeds und Wiederholungszahlen vergleichen;
- Momententwicklung mit vorhandenen analytischen Lösungen vergleichen;
- Nachbearbeitungsbilder erzeugen.

Hauptklassen:

| Klasse | Funktion |
| --- | --- |
| `CaseConfig` | Definiert physikalisches Problem: Dimension, Kernfunktion, Prozesstyp, Zeitgitter, analytische Lösungsparameter, Anfangspartikelmaßstab. |
| `WMCPBEVariantConfig` | Definiert eine WMCPBE-Parametergruppe, inklusive Wiederholungszahl, Zufallsseed, maximale Ereigniszahl und solver-Attributüberschreibung. |
| `GranulationValidationConfig` | Sammelt case und mehrere WMCPBE variants. |
| `GranulationValidationRunner` | Lässt alle WMCPBE variants laufen und erzeugt analytische Lösung. |
| `ValidationResult` | Speichert Zeit, Anfangszustand und Ergebnisse aller Methoden. |
| `ValidationPlotter` | Plottet Momente und Gesamtvolumen. |

## 6. Nutzung von `simple_validation.py`

`simple_validation.py` ist ein minimales Beispiel und zeigt, wie case und mehrere WMCPBE-Parametergruppen konfiguriert werden.

Grundstruktur:

```python
case = CaseConfig(...)

wmcpbe_variants = [
    WMCPBEVariantConfig(...),
    WMCPBEVariantConfig(...),
]

config = GranulationValidationConfig(
    case=case,
    wmcpbe_variants=wmcpbe_variants,
    verbose=False,
)

result = GranulationValidationRunner(config).run()
plotter = ValidationPlotter(result)
plotter.plot_all_moments(relative=True, include_total_volume=False)
plotter.show()
```

### 6.1 Wichtige Parameter von `CaseConfig`

| Parameter | Bedeutung |
| --- | --- |
| `dim` | Dimension, unterstützt `1` oder `2`. |
| `kernel` | Typ des Agglomerationskerns, Wrapper-Skript unterstützt momentan `"const"` und `"sum"`. |
| `process` | Prozesstyp: `"agglomeration"`, `"breakage"`, `"mix"`. |
| `t_vec` | Ausgabe-/Speicher-Zeitgitter. |
| `x` | Charakteristische Partikelgröße zur Konstruktion beispielhafter Anfangspartikelvolumina. |
| `beta0` | Analytische Lösung und Agglomerationskernparameter. |
| `g` | Scherrate oder verwandter Kernparameter. |
| `p1`, `p2` | Parameter der Bruchrate. In 2D sind `pl_P3/pl_P4` default gleich `p1/p2`. |
| `pl_v`, `pl_q` | Parameter der Fragmentverteilung. |
| `initial_number_density` | Anfangs-Zahldichte, Skript berechnet damit `Vc`. |
| `initial_total_weight` | Gesamtgewicht realer Partikel, vertreten durch Anfangs-Rechenpartikel. |

### 6.2 Wichtige Parameter von `WMCPBEVariantConfig`

| Parameter | Bedeutung |
| --- | --- |
| `name` | Name in Legende und Ergebnis-Dictionary. |
| `repeats` | Anzahl Wiederholungsrechnungen. Für statistischen Vergleich empfohlen größer als 1. |
| `base_seed` | Basis-Zufallsseed. |
| `maxiter` | Maximale Ereigniszahl pro Wiederholung. |
| `enabled` | Ob diese Parametergruppe aktiviert ist. |
| `attrs` | Parameter-Dictionary, das auf `MCPBESolver` überschrieben wird. |

`attrs` ist die meist geänderte Stelle. Zum Beispiel reconstruction schließen:

```python
attrs={
    "recon_enable": False,
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

reconstruction aktivieren:

```python
attrs={
    "recon_enable": True,
    "recon_method": "4PMC",
    "recon_N_max": 4000,
    "recon_bins": 30,
    "recon_RS_target": 1000,
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

batch-wise Stärke ändern:

```python
attrs={
    "break_dW_max": 10.0,
    "agg_dW_max": 20.0,
}
```

Wenn man möglichst nahe an Verarbeitung einzelner realer Ereignisse kommen will:

```python
attrs={
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

## 7. Anfangspartikel-Eingabe in validation

Die granulation-Version benutzt momentan explizite Partikeleingabe:

```python
V_flat + W_init + Vc
```

Das ist auch die für externe Entwickler am leichtesten verständliche und änderbare Form.

### 7.1 Eingabestelle

Default-Anfangspartikel sind definiert in:

```python
GranulationValidationRunner.build_example_initial_particles()
```

Diese Methode gibt zurück:

```python
InitialParticleState(
    Vc=...,
    V_flat=...,
    W_init=...,
)
```

Danach wird in `run_wmcpbe_variant()` übergeben an:

```python
solver.solve_repeats(
    init_Vc=False,
    Vc=initial_state.Vc,
    V_flat=initial_state.V_flat,
    W_init=initial_state.W_init,
)
```

### 7.2 Format von `V_flat`

`V_flat` muss ein zweidimensionales Array sein:

```python
V_flat.shape == (dim + 1, N_particles)
```

1D-Fall:

```python
V_flat[0, k] = Volumen des k-ten Partikels
V_flat[1, k] = Gesamtvolumen des k-ten Partikels
```

Da 1D nur eine Komponente hat:

```python
V_flat[1, :] = V_flat[0, :]
```

2D-Fall:

```python
V_flat[0, k] = Volumen von Komponente 1 im k-ten Partikel
V_flat[1, k] = Volumen von Komponente 2 im k-ten Partikel
V_flat[2, k] = Gesamtvolumen des k-ten Partikels = V_flat[0, k] + V_flat[1, k]
```

### 7.3 Format von `W_init`

`W_init` ist ein eindimensionales Array:

```python
W_init.shape == (N_particles,)
```

Bedeutung:

```python
W_init[k] = Anzahl realer Partikel, die vom k-ten Rechenpartikel vertreten wird
```

Zum Beispiel 4 Rechenpartikel vertreten verschiedene Mengen realer Partikel:

```python
W_init = np.array([40000.0, 20000.0, 20000.0, 20000.0])
```

### 7.4 Bedeutung von `Vc`

`Vc` ist Kontrollvolumen. Bei Moment-Nachbearbeitung wird damit normalisiert:

```python
mu00(0) = sum(W_init) / Vc
```

Im aktuellen Beispiel:

```python
Vc = initial_total_weight / initial_number_density
```

Deshalb kontrolliert `initial_number_density` das anfängliche `mu00`.

### 7.5 Aktuelle Beispielregeln

1D-Beispiel:

```python
component_volumes = np.array([[v0, 2*v0, 4*v0]])
fractions = np.array([0.60, 0.30, 0.10])
```

2D-Beispiel:

```python
component_volumes = np.array([
    [v0, 2*v0, v0, 2*v0],
    [v0, v0, 2*v0, 2*v0],
])
fractions = np.array([0.40, 0.20, 0.20, 0.20])
```

Das Skript konvertiert `fractions` zu Gewichten:

```python
W_init = initial_total_weight * fractions
```

### 7.6 Eigene Eingabe definieren, z.B. bekannte PSD

Wenn schon eine PSD oder externe Partikelverteilung vorhanden ist, empfohlen diese Methode umzuschreiben:

```python
build_example_initial_particles()
```

Allgemeine Schritte:

1. PSD in mehrere repräsentative Partikel diskretisieren.
2. Für jedes repräsentative Partikel Komponentenvolumen definieren.
3. Nach number fraction oder volume fraction der PSD `W_init` berechnen.
4. `V_flat` zusammenbauen.
5. Nach gewünschter Anfangs-Zahldichte `Vc` definieren.
6. `InitialParticleState` zurückgeben.

Pseudocode:

```python
def build_example_initial_particles(self) -> InitialParticleState:
    case = self.config.case

    # 1. Eigene PSD-Diskretisierungspunkte des Benutzers
    particle_volumes = ...
    particle_weights = ...

    # 2. V_flat zusammenbauen
    V_flat = np.zeros((case.dim + 1, particle_volumes.shape[1]))
    V_flat[:case.dim, :] = particle_volumes
    V_flat[-1, :] = np.sum(particle_volumes, axis=0)

    # 3. Kontrollvolumen definieren
    Vc = np.sum(particle_weights) / desired_number_density

    return InitialParticleState(
        Vc=Vc,
        V_flat=V_flat,
        W_init=particle_weights,
    )
```
