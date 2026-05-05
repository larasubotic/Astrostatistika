
import numpy as np
import matplotlib.pyplot as plt

## ZADATAK 1 #####################################################################################################################################

l, I = np.loadtxt('/home/laras/Faks/NUM/domaci 2/Spektar Sunca.txt', delimiter=',', skiprows=2, unpack=True) # ucitavamo dokument sa podacima

# plotujemo podatke da dobijemo grafik raspodele suncevog zracenja

plt.plot(l, I, linewidth=1, color="blue")
plt.xlabel("Talasna duzina [nm]")
plt.ylabel("Intenzitet zracenja [W/m²/nm]")
plt.title("Spektar Suncevog zracenja")
plt.show()

# trazimo na kojim talasnim duzinama se nalaze max i min vrednosti intenziteta zracenja
# pisemo kao np.argmin jer trazimo indeks talasne duzine na kojoj je min vrednost I, a ne kolika je vrednost min I

lmin = l[np.argmin(I)]
lmax = l[np.argmax(I)]

# stampamo rezultate za min i max vrednost 

print("Maksimum intenziteta zracenja se nalazi na ", lmax, "nm.")
print("Minimum intenziteta zracenja se nalazi na ", lmin, "nm.")


## ZADATAK 2 ##########################################################################################################################################################

tacnost = 1e-8
M = np.linspace(0, 2*np.pi, 100) # pravimo niz od 100 clanova izmedju 0 i 2pi
e = [0.05, 0.2, 0.5] # biramo tri vrednosti za ekscentricitet
M, e = np.meshgrid(M, e) # pravimo matricu M i e 3x100, tako da bi sve kombinacije mogle odmah da se racunaju

# definisemo keplerovu jednacinu 

def kep(M, e, tacnost=1e-8):

    E = M # definisemo pocetnu vrednost ekscentricne anomalije
    delta = np.ones_like(E) * 2 * tacnost # pravimo matricu iste velicine kao i E, ali je svaka vrednost postavljena sa 2*tacnost, pri cemu delta cuva promene u svakoj iteraciji
    br = 0 # krecemo sa 0 iteracija

    # Njutnova metoda 
    while np.any(np.abs(delta)>tacnost): # petlja radi dok delta ne bude manje od tacnosti 
        br += 1 # svaka iteracija se povecava za jedan
        fun = E - e*np.sin(E) - M # definisemo funkciju i njen izvod
        funprim = 1 - e*np.cos(E)
        delta = fun/funprim # kako nam se menja delta
        E = E - delta # kako nam se menja vrednost ekscentricne anomalije s promenom delte
    return E, br # kada je delta dovoljno malo, prestajemo sa iteracijama i ispisujemo konacne vrednosti

E, br = kep(M, e, tacnost) # vracamo pocetnu funkciju sa svim vrednostima M i e, ali sa novom preciznoscu koju smo dobili njutnovom metodom

# stampamo rezultate E za sve vrednosti M po svim e, tako da je e sa dve decimale, M sa 3, a E sa 6 decimala; kao i broj iteracija koji je bio potreban za datu tacnost

print("Rezultati (E) za različite e i M vrednosti:\n")
for i in range(e.shape[0]):
    print(f"e = {e[i,0]:.2f}:")
    for j in range(M.shape[1]):
        print(f"  M = {M[i,j]:.3f}  -->  E = {E[i,j]:.6f}")
    print()

print(f"Ukupan broj iteracija: {br}")

# crtamo grafik zavisnosti ekscentricne od srednje anomalije

plt.figure(figsize=(8,6))
for i, e_vr in enumerate([0.05, 0.2, 0.5]):
    plt.plot(M[i], E[i], label=f"e = {e_vr}") # za svaku vrednost ekscentriciteta uzima sve vrednosti M i E

plt.title("Zavisnost ekscentrične anomalije E od srednje anomalije M")
plt.xlabel("M (rad)")
plt.ylabel("E (rad)")
plt.legend()
plt.grid(True)
plt.show()






