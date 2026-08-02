"use strict";

const fs = require("fs");
const path = require("path");
const { parse } = require(path.resolve(
  "outputs/mechanistic_propagation_revised/_node_deps/node_modules/csv-parse"
));
const parquet = require(path.resolve(
  "outputs/mechanistic_propagation_revised/_node_deps/node_modules/parquetjs-lite"
));

const OUTPUT_DIR = path.resolve("outputs/mechanistic_propagation_revised");
const INPUT_CSV = path.join(OUTPUT_DIR, "sample_level_layerwise_traces.csv");
const OUTPUT_PARQUET = path.join(OUTPUT_DIR, "sample_level_layerwise_traces.parquet");
const TEMP_PARQUET = `${OUTPUT_PARQUET}.tmp`;

const stringFields = [
  "trace_schema_version", "cohort_id", "cohort_kind", "model_key", "model",
  "semantic_architecture", "attack", "attack_norm", "layer", "layer_display_name",
  "point_role", "tensor_name", "hook_side", "tensor_shape",
];
const integerFields = [
  "seed", "dataset_index", "class_label", "dimensionality",
  "clean_prediction", "adversarial_prediction",
];
const doubleFields = [
  "epsilon", "layer_order", "clean_margin", "delta_l2", "delta_rms", "gain_rms",
  "projected_harm", "normalized_harm", "residual_scale",
  "classical_to_quantum_displacement_ratio", "raw_classical_to_quantum_displacement_ratio",
  "quantum_norm_share_of_branch_sum", "classical_norm_share_of_branch_sum",
  "quantum_to_fused_norm_ratio", "classical_to_fused_norm_ratio",
  "quantum_projection_share_of_fused", "classical_projection_share_of_fused",
  "residual_reconstruction_relative_error",
];

const schemaDefinition = {};
for (const name of stringFields) schemaDefinition[name] = { type: "UTF8" };
for (const name of integerFields) schemaDefinition[name] = { type: "INT64" };
schemaDefinition["small_margin_lt_1e-6"] = { type: "BOOLEAN" };
for (const name of doubleFields) {
  schemaDefinition[name] = { type: "DOUBLE", optional: true, compression: "GZIP" };
}
const schema = new parquet.ParquetSchema(schemaDefinition);

function convert(row) {
  const converted = {};
  for (const name of stringFields) converted[name] = row[name];
  for (const name of integerFields) converted[name] = Number.parseInt(row[name], 10);
  converted["small_margin_lt_1e-6"] = String(row["small_margin_lt_1e-6"]).toLowerCase() === "true";
  for (const name of doubleFields) {
    const value = Number(row[name]);
    if (row[name] !== "" && row[name] !== undefined && Number.isFinite(value)) converted[name] = value;
  }
  return converted;
}

async function main() {
  if (fs.existsSync(TEMP_PARQUET)) fs.rmSync(TEMP_PARQUET);
  const writer = await parquet.ParquetWriter.openFile(schema, TEMP_PARQUET, {
    rowGroupSize: 4096,
  });
  let count = 0;
  const parser = fs.createReadStream(INPUT_CSV).pipe(parse({ columns: true, bom: true }));
  for await (const row of parser) {
    await writer.appendRow(convert(row));
    count += 1;
    if (count % 10000 === 0) process.stdout.write(`rows=${count}\n`);
  }
  await writer.close();
  if (fs.existsSync(OUTPUT_PARQUET)) fs.rmSync(OUTPUT_PARQUET);
  fs.renameSync(TEMP_PARQUET, OUTPUT_PARQUET);

  const reader = await parquet.ParquetReader.openFile(OUTPUT_PARQUET);
  const cursor = reader.getCursor();
  let verified = 0;
  while (await cursor.next()) verified += 1;
  await reader.close();
  if (verified !== count) throw new Error(`Parquet row-count mismatch: ${verified} != ${count}`);
  process.stdout.write(`verified_rows=${verified}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
