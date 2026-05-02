# Per-(model, dataset) training batch size on 1xGH200 (96 GiB), AR(x2) step.
# Linearly scaled from the H200 (140 GiB) probe by 96/140 ≈ 0.686, since
# activation memory dominates and scales linearly in batch.
# Resolution is the only dataset-side knob that matters for memory; in/out
# channel counts barely move the needle since hidden activations dominate.
# See scripts/probe_memory*.sbatch for the underlying H200 probe.
#
# Usage:
#     source scripts/batch_sizes.sh
#     bs=$(fots_batch_size zinnia mickelin)

declare -gA FOTS_DATA_RESOLUTION=(
    [planetswe]=256x512
    [galewsky]=256x512
    [cahn_hilliard]=256x512
    [shock_caps]=256x512
    [mickelin]=128x256
    [swe_th]=64x128
)

declare -gA FOTS_BATCH_SIZE=(
    # 256x512 (probe_memory.sbatch, PlanetSWE shape)
    [fno:256x512]=27
    [sfno:256x512]=26
    [flower:256x512]=22
    [zinnia:256x512]=9
    [dahlia:256x512]=9
    [dandelion:256x512]=9
    [local_r_transformer:256x512]=7
    [local_s2_transformer:256x512]=4

    # 128x256 (probe_memory_mickelin.sbatch)
    [fno:128x256]=98
    [sfno:128x256]=98
    [flower:128x256]=76
    [zinnia:128x256]=38
    [dahlia:128x256]=38
    [dandelion:128x256]=43
    [local_r_transformer:128x256]=27
    [local_s2_transformer:128x256]=13
)

fots_batch_size() {
    local model="$1" dataset="$2"
    local res="${FOTS_DATA_RESOLUTION[$dataset]:-}"
    if [ -z "$res" ]; then
        echo "ERROR: unknown dataset '$dataset' — add to FOTS_DATA_RESOLUTION in scripts/batch_sizes.sh" >&2
        return 1
    fi
    local bs="${FOTS_BATCH_SIZE[$model:$res]:-}"
    if [ -z "$bs" ]; then
        echo "ERROR: no batch size for ($model, $res) — run scripts/probe_memory*.sbatch and add to FOTS_BATCH_SIZE in scripts/batch_sizes.sh" >&2
        return 1
    fi
    echo "$bs"
}
