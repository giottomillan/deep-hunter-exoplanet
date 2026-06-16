from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "images_kepler_style"
)


N_SAMPLES = 500
N_POINTS = 1000

SEED = 42


def planet_curve():

    x = np.linspace(
        0,
        1,
        N_POINTS
    )

    flux = np.ones(
        N_POINTS
    )


    # stellar variability

    flux += (
        np.random.uniform(
            0.002,
            0.015,
        )
        *
        np.sin(
            2
            *
            np.pi
            *
            np.random.uniform(1, 5)
            *
            x
        )
    )


    center = np.random.uniform(
        0.3,
        0.7,
    )

    width = np.random.uniform(
        0.01,
        0.05,
    )

    depth = np.random.uniform(
        0.005,
        0.035,
    )


    transit = (
        depth
        *
        np.exp(
            -0.5
            *
            (
                (x-center)
                /
                width
            )
            ** 2
        )
    )


    flux -= transit


    flux += np.random.normal(
        0,
        0.01,
        N_POINTS,
    )


    return flux



def false_positive_curve():

    x = np.linspace(
        0,
        1,
        N_POINTS
    )

    flux = np.ones(
        N_POINTS
    )


    choice = np.random.choice(
        [
            "noise",
            "variable",
            "dip"
        ]
    )


    if choice == "noise":

        flux += np.random.normal(
            0,
            0.02,
            N_POINTS,
        )


    elif choice == "variable":

        flux += (
            np.random.uniform(
                0.01,
                0.04
            )
            *
            np.sin(
                2
                *
                np.pi
                *
                np.random.uniform(
                    2,
                    8
                )
                *
                x
            )
        )


    else:

        center = np.random.uniform(
            0.2,
            0.8,
        )

        width = np.random.uniform(
            0.02,
            0.08,
        )

        depth = np.random.uniform(
            0.005,
            0.03,
        )

        flux -= (
            depth
            *
            np.exp(
                -0.5
                *
                (
                    (x-center)
                    /
                    width
                )
                **2
            )
        )


    flux += np.random.normal(
        0,
        0.012,
        N_POINTS
    )


    return flux



def save_image(
    flux,
    path
):

    fig, ax = plt.subplots(
        figsize=(6,3),
        dpi=100
    )


    ax.plot(
        flux,
        color="black",
        linewidth=0.8,
    )


    # Kepler-like vertical artifacts

    for _ in range(
        np.random.randint(
            5,
            20
        )
    ):

        position = np.random.randint(
            0,
            N_POINTS
        )

        ax.axvline(
            position,
            linewidth=0.3,
            alpha=0.4,
        )


    ax.axis(
        "off"
    )

    ax.set_ylim(
        0.88,
        1.08
    )

    plt.tight_layout(
        pad=0
    )


    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
    )


    plt.close(fig)



def main():

    np.random.seed(
        SEED
    )


    folders = {
        "planet": planet_curve,
        "false_positive": false_positive_curve,
    }


    for name, generator in folders.items():

        folder = OUTPUT_DIR / name

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


        for i in tqdm(
            range(N_SAMPLES),
            desc=name
        ):

            flux = generator()

            save_image(
                flux,
                folder / f"{name}_{i}.png"
            )


    print(
        "Kepler style dataset created"
    )



if __name__ == "__main__":
    main()