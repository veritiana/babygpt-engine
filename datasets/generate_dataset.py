import random

# Fixná báza faktov (Stálosť)
facts = [
    "Slnko je hviezda v centre našej slnečnej sústavy a vyžaruje svetlo.",
    "Planéta Merkúr je najbližšia k Slnku a má veľmi tenkú atmosféru.",
    "Planéta Venuša je druhá v poradí a je najteplejšou planétou sústavy.",
    "Planéta Zem je naším domovom a ako jediná má potvrdený život.",
    "Planéta Mars je známa ako červená planéta kvôli oxidu železa.",
    "Planéta Jupiter je najväčší plynný obor s mnohými mesiacmi.",
    "Planéta Saturn má najvýraznejšie prstence zložené z ľadu a prachu.",
    "Planéta Urán je ľadový obor a rotuje na boku s veľkým sklonom.",
    "Planéta Neptún je najvzdialenejšia od Slnka a bičujú ju silné vetry.",
    "Mesiac je prirodzený satelit Zeme a ovplyvňuje príliv a odliv.",
    "Čierna diera je oblasť vesmíru, z ktorej neunikne ani svetlo.",
    "Mliečna dráha je galaxia, v ktorej sa nachádza naša slnečná sústava.",
    "Svetelný rok je jednotka vzdialenosti, ktorú prejde svetlo za rok.",
    "Gravitácia je sila, ktorá drží planéty na ich obežných dráhach.",
    "Asteroidy sú malé kamenné telesá obiehajúce okolo Slnka.",
    "Kométy sú telesá z ľadu a prachu, ktoré vytvárajú jasný chvost.",
    "Supernova je gigantický výbuch hviezdy na konci jej života.",
    "Exoplanéta je planéta, ktorá obieha inú hviezdu než naše Slnko.",
    "Astronauti cestujú do vesmíru v kozmických lodiach a raketách.",
    "Teleskop je prístroj používaný na pozorovanie vzdialených hviezd.",
    "Vesmír sa neustále rozpína od momentu veľkého tresku."
]

target_size_mb = 1.0
file_path = "dataset_sk.txt"

def generate():
    current_size = 0
    with open(file_path, "w", encoding="utf-8") as f:
        while current_size < target_size_mb * 1024 * 1024:
            # Držíme sa faktov, len meníme poradie
            line = random.choice(facts) + "\n"
            f.write(line)
            current_size += len(line.encode("utf-8"))

if __name__ == "__main__":
    generate()
    print(f"Dataset generated: {file_path} (Size: ~1 MB)")