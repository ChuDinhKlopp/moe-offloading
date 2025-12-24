#!/usr/bin/bash
DUMP_DIR=./dump
LOG_DIR=./log

SERVER_DIR=server
BENCH_DIR=bench
DEVICE_DIR=device
OBS_DIR=observability
TIMING_DIR=timing
EXPERT_ACTIVATION_DIR=expert_activation_pattern

rm -rf $DUMP_DIR/$SERVER_DIR/*
rm -rf $DUMP_DIR/$BENCH_DIR/*
rm -rf $DUMP_DIR/$OBS_DIR/*

rm -rf $LOG_DIR/$BENCH_DIR/*
rm -rf $LOG_DIR/$DEVICE_DIR/*
rm -rf $LOG_DIR/$OBS_DIR/*
rm -rf $LOG_DIR/$TIMING_DIR/*
# rm -rf $LOG_DIR/$EXPERT_ACTIVATION_DIR/*
