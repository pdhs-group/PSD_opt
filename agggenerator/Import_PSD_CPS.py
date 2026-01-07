# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:18:40 2025

@author: em4724

Datenstruktur von bereits ausgewerteten Daten: 
    Lineare Intervalle
    Erste Spalte:
        erste Zeile minimal Durchmesser
        letzte Zeile maximal Durchmesser --> Threshold Variable um Agglomerate bei Bedarf abzuschneiden
        Intervall-Breite konstant Differenz zwischen zwei Zeilen
    
    Zweite Spalte:
        normierte q3-PSD --> Summe aus Spalte 

Algo zur Integration:
    Erste Zeile Q3[0]=q3[0]*deltaX/Summe_q3
    for schleife über alle zeiln von min_d+1 bis max_d
        Q3_i = Summe_q3(von 0 bis i)*deltax/Summe_q3
        
Umrechnung Q3 --> Q0:
    Anfangen bei q3
    Berechnung arithmetisches Mittel q3 nach Gl 2.48 Stieß (S. 41) --> [M0,r = 1; Normierungsbedingung] Bullshit
    Berechnung -3/3tes Moment der q3-Verteilung nach 2.110 S.64
Vorgabe Anzahl an Aggregaten die es zu erzeugen gilt --> Maximale Punkte an Daten
Q0[i] = So viele Aggregate mussen bis dahin erzeugt worden sein, Erzeugung im Größenbereich i 
tatsächliche Größe kann einfach mit randInt gewählt werden :D (spaxxerino) besser: gleichmäßige Verteilung durch linspacen des Intervalls mit Anzahl Aggregate in Intervall

q0 steht. allerdings ist PSD zu feingranular mit 200 bins
rebinning auf N --> Maximaler Humbug; Hier entweder CutOff notwendig oder log_spacing; Guter Cuttoff schwierig zu programmieren v.A. für bimodale GL; log-Spacing eleganter
Nummer der Aggregate auf jedem Intervall definiert
x_min und x_Max aus jedem Intervall berechnen
lin_space auf jedem Intervall mit Nummer der Aggregate --> Jedem Aggregat wird ein konkreter Durchmesser zugewiesen

Kleinster Durchmesser berechnen; Startwert --> x_int[0]
Array in Länge Summe[fractions] 
Dann For-Schleife: Distanz x_Int[i] --> X_Int[i-1] für i >= 1
    Intervall zwischen x_Int[i] - X_Int[i-1] = delta X
    Delta X/q_agg_diss[i]
    Array vollklatschen mit jeweils [fractions] Einträgen; --> Jeder Eintrag ist der Durchmesser eines Aggregates den es 
    Array 

Prüfen ob log/ln überall korrekt verwendet wurde

Größe und Anzahl Pixel korrelieren
Dann einfach Schleife um Haorans Skript schreiben

Konvertieren in class PSD?
"""



import numpy as np
import os
from glob import glob
import pandas as pd

#will move to main program 
def parameters():
    N_bins = 20             #Number of bins the original data will be reduced to
    N_aggs = 1000            #number of aggregates that will be generated
    cell_size = 0.01       #Cell size in micrometers
    return N_bins, N_aggs, cell_size



def main():
    #if PSD is exported from origin as .dat
    path = os.path.join('input/PSD_agg','*.dat') #For now, coded with a single file as input in subfolder 
    d3_cent, q3_agg = read_origin_dat(path)
    N_bins, N_aggs, cell_size = parameters()
    list_np, x_dis, q0_agg_dis, psd_to_gen = transform_PSD(d3_cent, q3_agg, N_bins, N_aggs, cell_size)
    # Save directly into a file
    #output_path = "psd_example.npz"
    return list_np, x_dis, q0_agg_dis, psd_to_gen 

def read_origin_dat(path):
    # Load the file as a CSV with comma separation, skipping the header row;
    DatFiles = sorted(glob(path))
    df = pd.read_csv(DatFiles[0], encoding="latin1")

    d3_cent = df.iloc[:, 0].values
    q3_agg  = df.iloc[:, 1].values

    return d3_cent, q3_agg

def transform_PSD(d3_cent, q3_agg, N_bins, N_aggs, cell_size):
    #d3_mean = np.sum(d3_cent[:]*q3_agg[:]/np.sum(q3_agg))              Hanebüchen, da ist ein Fehler drin
    #d3_mean = np.sum(d3_cent[:]*q3_agg[:]*(d3_cent[2]-d3_cent[1]))
    
    # =============================================================================
    # Recalculate the raw lin-spaced q3-PSD to a discretized q0 log-spaced PSD with N_bins bin
    # =============================================================================
    M_neg3_3 = np.sum(d3_cent[:]**(-3)*q3_agg[:]*(d3_cent[2]-d3_cent[1]))  #Calculation of -3/3tes Moment of q3-PSD, compare eq 2.110 S.64 Stieß
    q0_agg = d3_cent[:]**(-3)*q3_agg[:]/M_neg3_3                            #calculation q0, compare eq 2.119 P 65
    x_dis = np.geomspace(d3_cent[0],d3_cent[-1],N_bins)                     #recalculation of the bins in log-space 
    q0_agg_dis = np.interp(x_dis, d3_cent, q0_agg)                          #recalculation discretized q0
    
    
    # =============================================================================
    # Normalizing q0_agg_dis with N_agg
    # =============================================================================
    #q0_agg_dis_N_aggs = (q0_agg_dis[:]/np.sum(q0_agg_dis[:]))*N_aggs                   #renormalizing the discretized q0; might be bullshit because deltaX 
    fractions = np.rint(q0_agg_dis[:]*x_dis[:]/np.sum(q0_agg_dis[:]*x_dis[:])*N_aggs)                   #Eq 2.41b p34; Number of aggs in each bin, rounded to the nearest int
    fractions_int = fractions.astype(int)
    
    #Calculation of upper limit of the intervalls of discretized PSD
    x_Int = np.zeros([len(x_dis),1])
    for i in range(len(x_dis)-1) :
        x_Int[i] = (x_dis[i+1]-x_dis[i])/np.log((x_dis[i+1]/x_dis[i]))                #calculating upper limits of the intervals in log space
    
    x_Int[-1] = x_dis[-1]+(x_dis[-1]-x_Int[-2])                                        #Last entry of upper limit calculated manually
    x_0 = x_dis[0]-(x_Int[0]-x_dis[0])                                                  #Smallest particle size
    
    # ==============================================================================
    # Between the intervall thresholds calculate [fractions] linear spaces --> number of aggregates in size intervall conserved
    # =============================================================================
    parts = []
    if fractions[0] != 0:
        parts.append(np.linspace(x_0, x_Int[0], fractions[0]))

    for i in range(1, len(x_dis)):
        if fractions_int[i] != 0:
            parts.append(np.linspace(x_Int[i-1], x_Int[i], fractions_int[i], endpoint=False))
    psd_to_gen = np.concatenate(parts)
    list_np = np.rint(psd_to_gen[:]**2/cell_size**2)            #Calculate the number of pixels per Aggregate with specified cell size
   
    return list_np, x_dis, q0_agg_dis, psd_to_gen 


if __name__ == "__main__":
    list_np, x_dis, q0_agg_dis, psd_to_gen = main()