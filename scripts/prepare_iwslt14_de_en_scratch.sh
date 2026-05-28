#!/usr/bin/env bash
set -euo pipefail

# Canonical IWSLT'14 De->En preprocessing without cloning fairseq.
#
# This reproduces the fairseq/Moses/subword-nmt text preparation pipeline:
#   - download de-en.tgz
#   - tokenize with Moses
#   - lowercase
#   - clean corpus
#   - create the fairseq train/valid/test split
#   - learn 10k joint BPE
#   - apply BPE to train/valid/test
#   - pack a tarball for fast copy to SLURM_TMPDIR
#
# Usage examples:
#   SCRATCH=/network/scratch/$USER bash scripts/prepare_iwslt14_de_en_scratch.sh
#   SCRATCH=/network/scratch/$USER MOSES_ROOT=/path/to/mosesdecoder \
#     SUBWORD_NMT_ROOT=/path/to/subword-nmt bash scripts/prepare_iwslt14_de_en_scratch.sh
#
# Expected outputs:
#   $SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en/
#   $SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en.tar.gz
#
# Then inside a SLURM job:
#   tar -xzf $SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en.tar.gz -C $SLURM_TMPDIR
#   export IWSLT14_DATA_DIR=$SLURM_TMPDIR/iwslt14.tokenized.de-en

: "${SCRATCH:?SCRATCH must be set}"

SRC=de
TGT=en
LANG=de-en
BPE_TOKENS=10000
DATA_URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
ARCHIVE_NAME="de-en.tgz"

CACHE_ROOT="${SCRATCH}/iwslt14_de_en_cache"
RAW_ROOT="${CACHE_ROOT}/orig"
WORK_ROOT="${CACHE_ROOT}/work"
OUT_DIR="${CACHE_ROOT}/iwslt14.tokenized.de-en"
TMP_DIR="${WORK_ROOT}/tmp"
ARCHIVE_PATH="${RAW_ROOT}/${ARCHIVE_NAME}"
PACKED_DATASET="${CACHE_ROOT}/iwslt14.tokenized.de-en.tar.gz"

mkdir -p "${RAW_ROOT}" "${WORK_ROOT}" "${TMP_DIR}" "${OUT_DIR}"

if [[ -z "${MOSES_ROOT:-}" ]]; then
  MOSES_ROOT="${CACHE_ROOT}/tools/mosesdecoder"
fi
if [[ -z "${SUBWORD_NMT_ROOT:-}" ]]; then
  SUBWORD_NMT_ROOT="${CACHE_ROOT}/tools/subword-nmt"
fi

if [[ ! -d "${MOSES_ROOT}/scripts" ]]; then
  echo "[IWSLT14] Moses not found at ${MOSES_ROOT}. Cloning a local copy..."
  mkdir -p "$(dirname "${MOSES_ROOT}")"
  git clone https://github.com/moses-smt/mosesdecoder.git "${MOSES_ROOT}"
fi

if [[ ! -f "${SUBWORD_NMT_ROOT}/subword_nmt/learn_bpe.py" ]]; then
  echo "[IWSLT14] subword-nmt not found at ${SUBWORD_NMT_ROOT}. Cloning a local copy..."
  mkdir -p "$(dirname "${SUBWORD_NMT_ROOT}")"
  git clone https://github.com/rsennrich/subword-nmt.git "${SUBWORD_NMT_ROOT}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

TOKENIZER="${MOSES_ROOT}/scripts/tokenizer/tokenizer.perl"
LOWERCASE="${MOSES_ROOT}/scripts/tokenizer/lowercase.perl"
CLEAN="${MOSES_ROOT}/scripts/training/clean-corpus-n.perl"
LEARN_BPE="${SUBWORD_NMT_ROOT}/subword_nmt/learn_bpe.py"
APPLY_BPE="${SUBWORD_NMT_ROOT}/subword_nmt/apply_bpe.py"

for f in "${TOKENIZER}" "${LOWERCASE}" "${CLEAN}" "${LEARN_BPE}" "${APPLY_BPE}"; do
  [[ -f "${f}" ]] || { echo "[IWSLT14] Missing dependency file: ${f}"; exit 1; }
done

download_file() {
  local url="$1"
  local output_path="$2"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${output_path}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output_path}"
  else
    echo "[IWSLT14] Either wget or curl is required to download ${url}." >&2
    exit 1
  fi
}

if [[ -f "${OUT_DIR}/train.${SRC}" && -f "${OUT_DIR}/train.${TGT}" && -f "${OUT_DIR}/valid.${SRC}" \
      && -f "${OUT_DIR}/valid.${TGT}" && -f "${OUT_DIR}/test.${SRC}" && -f "${OUT_DIR}/test.${TGT}" \
      && -f "${OUT_DIR}/code" && -f "${PACKED_DATASET}" ]]; then
  echo "[IWSLT14] Processed dataset and tarball already exist; nothing to do."
  echo "[IWSLT14] Processed text dataset : ${OUT_DIR}"
  echo "[IWSLT14] Packed dataset tarball : ${PACKED_DATASET}"
  exit 0
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "[IWSLT14] Downloading ${DATA_URL}"
  download_file "${DATA_URL}" "${ARCHIVE_PATH}"
else
  echo "[IWSLT14] Reusing cached archive ${ARCHIVE_PATH}"
fi

if [[ ! -d "${RAW_ROOT}/${LANG}" ]]; then
  echo "[IWSLT14] Extracting ${ARCHIVE_PATH}"
  tar -xzf "${ARCHIVE_PATH}" -C "${RAW_ROOT}"
else
  echo "[IWSLT14] Reusing extracted raw directory ${RAW_ROOT}/${LANG}"
fi

if [[ -f "${OUT_DIR}/train.${SRC}" && -f "${OUT_DIR}/train.${TGT}" && -f "${OUT_DIR}/valid.${SRC}" \
      && -f "${OUT_DIR}/valid.${TGT}" && -f "${OUT_DIR}/test.${SRC}" && -f "${OUT_DIR}/test.${TGT}" \
      && -f "${OUT_DIR}/code" ]]; then
  echo "[IWSLT14] Processed dataset already exists in ${OUT_DIR}; skipping preprocessing."
else
  echo "[IWSLT14] Preprocessing training split"
  for L in "${SRC}" "${TGT}"; do
    IN_FILE="${RAW_ROOT}/${LANG}/train.tags.${LANG}.${L}"
    TOK_FILE="${TMP_DIR}/train.tags.${LANG}.tok.${L}"

    grep -v '<url>' "${IN_FILE}" \
      | grep -v '<talkid>' \
      | grep -v '<keywords>' \
      | sed -e 's/<title>//g' \
            -e 's#</title>##g' \
            -e 's/<description>//g' \
            -e 's#</description>##g' \
      | perl "${TOKENIZER}" -threads 8 -l "${L}" > "${TOK_FILE}"
  done

  perl "${CLEAN}" -ratio 1.5 \
    "${TMP_DIR}/train.tags.${LANG}.tok" "${SRC}" "${TGT}" \
    "${TMP_DIR}/train.tags.${LANG}.clean" 1 175

  for L in "${SRC}" "${TGT}"; do
    perl "${LOWERCASE}" < "${TMP_DIR}/train.tags.${LANG}.clean.${L}" > "${TMP_DIR}/train.tags.${LANG}.${L}"
  done

  echo "[IWSLT14] Preprocessing valid/test XML files"
  for L in "${SRC}" "${TGT}"; do
    for XML_PATH in "${RAW_ROOT}/${LANG}"/IWSLT14.TED*.${L}.xml; do
      BASE_NAME="$(basename "${XML_PATH}" .xml)"
      OUT_PATH="${TMP_DIR}/${BASE_NAME}"
      grep '<seg id' "${XML_PATH}" \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
              -e 's/\s*<\/seg>\s*//g' \
        | perl -CS -pe "s/\\x{2019}/'/g" \
        | perl "${TOKENIZER}" -threads 8 -l "${L}" \
        | perl "${LOWERCASE}" > "${OUT_PATH}"
    done
  done

  echo "[IWSLT14] Creating canonical train/valid/test split"
  for L in "${SRC}" "${TGT}"; do
    awk '{ if (NR % 23 == 0) print $0; }' "${TMP_DIR}/train.tags.${LANG}.${L}" > "${TMP_DIR}/valid.${L}"
    awk '{ if (NR % 23 != 0) print $0; }' "${TMP_DIR}/train.tags.${LANG}.${L}" > "${TMP_DIR}/train.${L}"

    cat \
      "${TMP_DIR}/IWSLT14.TED.dev2010.de-en.${L}" \
      "${TMP_DIR}/IWSLT14.TEDX.dev2012.de-en.${L}" \
      "${TMP_DIR}/IWSLT14.TED.tst2010.de-en.${L}" \
      "${TMP_DIR}/IWSLT14.TED.tst2011.de-en.${L}" \
      "${TMP_DIR}/IWSLT14.TED.tst2012.de-en.${L}" \
      > "${TMP_DIR}/test.${L}"
  done

  echo "[IWSLT14] Learning joint ${BPE_TOKENS}-merge BPE"
  COMBINED_TRAIN="${TMP_DIR}/train.en-de"
  rm -f "${COMBINED_TRAIN}"
  cat "${TMP_DIR}/train.${SRC}" >> "${COMBINED_TRAIN}"
  cat "${TMP_DIR}/train.${TGT}" >> "${COMBINED_TRAIN}"

  "${PYTHON_CMD}" "${LEARN_BPE}" -s "${BPE_TOKENS}" < "${COMBINED_TRAIN}" > "${OUT_DIR}/code"

  echo "[IWSLT14] Applying BPE"
  for L in "${SRC}" "${TGT}"; do
    for SPLIT in train valid test; do
      "${PYTHON_CMD}" "${APPLY_BPE}" -c "${OUT_DIR}/code" < "${TMP_DIR}/${SPLIT}.${L}" > "${OUT_DIR}/${SPLIT}.${L}"
    done
  done
fi

if [[ -f "${PACKED_DATASET}" ]]; then
  echo "[IWSLT14] Packed dataset already exists at ${PACKED_DATASET}; keeping cached tarball."
else
  echo "[IWSLT14] Creating packed dataset ${PACKED_DATASET}"
  tar -czf "${PACKED_DATASET}" -C "${CACHE_ROOT}" "iwslt14.tokenized.de-en"
fi

echo "[IWSLT14] Done. Useful paths:"
echo "  Processed text dataset : ${OUT_DIR}"
echo "  Packed dataset tarball : ${PACKED_DATASET}"
echo
cat <<'MSG'
Example SLURM-side staging snippet:

  echo "Copying IWSLT14 de-en dataset"
  cd "$SLURM_TMPDIR"
  tar -xzf "$SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en.tar.gz"
  echo "Finished copying IWSLT14 data"
  echo "Dataset located in"
  echo "$SLURM_TMPDIR/iwslt14.tokenized.de-en"
  export IWSLT14_DATA_DIR="$SLURM_TMPDIR/iwslt14.tokenized.de-en"

MSG
