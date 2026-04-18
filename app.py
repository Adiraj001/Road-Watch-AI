from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template_string, request
from config import Config
from dashboard import mount_dashboard
from reporter import create_and_save_report
from storage import (
    get_all_potholes,
    get_counts,
    get_hourly_counts,
    get_severity_counts,
    get_status_counts,
    get_zone_counts,
    initialize_storage,
    mark_as_fixed,
    seed_dummy_data,
)
from yolo_detect import get_model_status

import subprocess
import sys
import os

import base64
import numpy as np
import cv2
from yolo_detect import detect_frame


# ══════════════════════════════════════════════════════════════════════
# HOME PAGE — Three.js 3D Command Center with Framer Motion
# ══════════════════════════════════════════════════════════════════════
HOME_PAGE_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RoadWatch AI — Command Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Bebas+Neue&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://unpkg.com/framer-motion@11/dist/framer-motion.js"></script>
  <style>
    :root {
      --saffron:   #FF9933;
      --saffron2:  #FF6B00;
      --green:     #00E676;
      --navy:      #0A0F1E;
      --navy2:     #0D1627;
      --navy3:     #111D35;
      --red:       #FF3D57;
      --blue:      #00B4FF;
      --text:      #C8D6E5;
      --text-dim:  #5A7A9A;
      --glass:     rgba(13,22,39,0.75);
      --glass-bd:  rgba(255,153,51,0.15);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow-x: hidden; }
    body {
      background: var(--navy);
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Tri-colour strip ── */
    .tristrip {
      position: fixed; top: 0; left: 0; right: 0; height: 4px; z-index: 9999;
      background: linear-gradient(90deg, #FF9933 0% 33%, #fff 33% 66%, #138808 66% 100%);
    }

    /* ── Three.js canvas ── */
    #bg-canvas {
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      z-index: 0; pointer-events: none;
    }

    /* ── Scan-line overlay ── */
    .scanlines {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 1;
      background: repeating-linear-gradient(
        0deg,
        transparent, transparent 2px,
        rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px
      );
      pointer-events: none;
    }

    /* ── Content ── */
    .page { position: relative; z-index: 2; min-height: 100vh; display: flex; flex-direction: column; }

    /* ── Header ── */
    .header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 18px 40px; margin-top: 4px;
      border-bottom: 1px solid rgba(255,153,51,0.2);
      backdrop-filter: blur(10px);
      background: rgba(10,15,30,0.8);
    }
    .brand-wrap { display: flex; align-items: center; gap: 16px; }
    .chakra { font-size: 32px; color: var(--saffron); animation: spin3d 12s linear infinite; display: inline-block; }
    @keyframes spin3d {
      0%   { transform: rotateY(0deg); }
      100% { transform: rotateY(360deg); }
    }
    .brand-name {
      font-family: 'Orbitron', sans-serif;
      font-size: 22px; font-weight: 900;
      letter-spacing: 4px; color: #fff;
      text-shadow: 0 0 20px rgba(255,153,51,0.5);
    }
    .brand-name em { color: var(--saffron); font-style: normal; }
    .brand-sub { font-size: 9px; letter-spacing: 3px; color: var(--text-dim); margin-top: 3px; }

    .header-right { display: flex; align-items: center; gap: 20px; }
    .live-badge {
      display: flex; align-items: center; gap: 8px;
      font-family: 'Orbitron', sans-serif;
      font-size: 10px; font-weight: 600; letter-spacing: 2px;
      color: var(--green);
      padding: 6px 14px;
      border: 1px solid rgba(0,230,118,0.3);
      border-radius: 4px;
      background: rgba(0,230,118,0.05);
    }
    .pulse-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: blink 1.4s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.8)} }

    .clock {
      font-family: 'Orbitron', sans-serif;
      font-size: 13px; font-weight: 500;
      color: var(--saffron); letter-spacing: 2px;
      text-shadow: 0 0 10px rgba(255,153,51,0.4);
    }

    /* ── Hero section ── */
    .hero {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      padding: 60px 40px 40px;
      text-align: center;
      position: relative;
    }
    .hero-eyebrow {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px; letter-spacing: 5px; color: var(--saffron);
      margin-bottom: 20px;
      opacity: 0; transform: translateY(20px);
    }
    .hero-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(56px, 8vw, 110px);
      letter-spacing: 8px; line-height: 0.95;
      color: #fff;
      text-shadow:
        0 0 40px rgba(255,153,51,0.3),
        0 0 80px rgba(255,153,51,0.1);
      opacity: 0; transform: translateY(30px);
    }
    .hero-title .accent { color: var(--saffron); }
    .hero-sub {
      font-size: 13px; letter-spacing: 2px; color: var(--text-dim);
      margin-top: 20px; max-width: 500px; line-height: 1.7;
      opacity: 0; transform: translateY(20px);
    }

    /* ── Glowing divider ── */
    .glow-line {
      width: 200px; height: 1px;
      background: linear-gradient(90deg, transparent, var(--saffron), transparent);
      margin: 30px auto;
      box-shadow: 0 0 10px var(--saffron);
      opacity: 0;
    }

    /* ── Stats grid ── */
    .stats-section { padding: 10px 40px 40px; }
    .stats-label {
      font-family: 'Orbitron', sans-serif;
      font-size: 10px; font-weight: 600; letter-spacing: 4px;
      color: var(--text-dim); text-align: center;
      margin-bottom: 24px;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px; max-width: 1100px; margin: 0 auto;
    }
    .stat-card {
      background: var(--glass);
      border: 1px solid var(--glass-bd);
      border-radius: 8px;
      padding: 24px 20px;
      text-align: center;
      position: relative; overflow: hidden;
      cursor: default;
      opacity: 0; transform: translateY(40px) scale(0.95);
      transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
      backdrop-filter: blur(12px);
    }
    .stat-card::before {
      content: ''; position: absolute;
      top: 0; left: 0; right: 0; height: 2px;
      background: var(--accent, var(--saffron));
      box-shadow: 0 0 12px var(--accent, var(--saffron));
    }
    .stat-card::after {
      content: ''; position: absolute;
      inset: 0; border-radius: 8px;
      background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.03), transparent 60%);
      pointer-events: none;
    }
    .stat-card:hover {
      transform: translateY(-6px) scale(1.02) rotateX(4deg);
      box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px rgba(255,153,51,0.1);
      border-color: rgba(255,153,51,0.3);
    }
    .stat-icon { font-size: 20px; margin-bottom: 10px; }
    .stat-label {
      font-size: 9px; letter-spacing: 2.5px;
      color: var(--text-dim); margin-bottom: 10px;
      font-family: 'Orbitron', sans-serif;
    }
    .stat-value {
      font-family: 'Orbitron', sans-serif;
      font-size: 32px; font-weight: 800;
      color: #fff;
      text-shadow: 0 0 20px var(--accent, rgba(255,153,51,0.5));
    }
    .stat-unit { font-size: 14px; color: var(--text-dim); margin-left: 3px; }

    /* ── Action cards ── */
    .actions-section { padding: 0 40px 60px; }
    .actions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px; max-width: 1100px; margin: 0 auto;
    }
    .action-card {
      background: var(--glass);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px; padding: 32px;
      text-decoration: none; color: inherit;
      position: relative; overflow: hidden;
      opacity: 0; transform: translateX(-30px);
      transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
      backdrop-filter: blur(12px);
      display: flex; flex-direction: column; gap: 16px;
    }
    .action-card:hover {
      transform: translateY(-8px);
      box-shadow: 0 30px 60px rgba(0,0,0,0.5),
                  0 0 30px rgba(var(--card-glow, 255,153,51), 0.15);
      border-color: rgba(var(--card-glow, 255,153,51), 0.3);
    }
    .action-card .card-bg {
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 0% 0%, rgba(var(--card-glow,255,153,51),0.06), transparent 60%);
      pointer-events: none;
    }
    .card-icon-wrap {
      width: 56px; height: 56px; border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 26px;
      background: rgba(var(--card-glow,255,153,51),0.1);
      border: 1px solid rgba(var(--card-glow,255,153,51),0.2);
    }
    .card-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 14px; font-weight: 700;
      letter-spacing: 2px; color: #fff;
    }
    .card-desc { font-size: 11px; color: var(--text-dim); line-height: 1.7; letter-spacing: 0.5px; }
    .card-arrow {
      margin-top: auto;
      font-family: 'Orbitron', sans-serif;
      font-size: 10px; letter-spacing: 3px;
      color: var(--saffron); display: flex; align-items: center; gap: 8px;
      transition: gap 0.3s ease;
    }
    .action-card:hover .card-arrow { gap: 16px; }
    .arrow-line {
      height: 1px; width: 30px;
      background: var(--saffron);
      box-shadow: 0 0 6px var(--saffron);
      transition: width 0.3s ease;
    }
    .action-card:hover .arrow-line { width: 50px; }

    /* ── Zone hotspot ticker ── */
    .ticker-wrap {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: rgba(10,15,30,0.95);
      border-top: 1px solid rgba(255,153,51,0.2);
      overflow: hidden; height: 36px; z-index: 100;
      display: flex; align-items: center;
    }
    .ticker-label {
      flex-shrink: 0; padding: 0 18px;
      font-family: 'Orbitron', sans-serif;
      font-size: 9px; font-weight: 700; letter-spacing: 3px;
      color: var(--saffron);
      border-right: 1px solid rgba(255,153,51,0.2);
      height: 100%; display: flex; align-items: center;
      background: rgba(255,153,51,0.05);
    }
    .ticker-track {
      display: flex; align-items: center;
      animation: ticker 22s linear infinite;
      white-space: nowrap; gap: 60px; padding-left: 40px;
    }
    @keyframes ticker { from{transform:translateX(0)} to{transform:translateX(-50%)} }
    .ticker-item {
      font-size: 11px; letter-spacing: 1.5px;
      display: flex; align-items: center; gap: 10px;
    }
    .ticker-item .zone { color: #fff; font-weight: 600; }
    .ticker-item .cnt {
      background: var(--red); color: #fff;
      padding: 1px 8px; border-radius: 3px;
      font-size: 10px; font-weight: 700;
    }
    .ticker-sep { color: rgba(255,153,51,0.3); font-size: 16px; }

    /* ── Footer ── */
    .footer {
      text-align: center; padding: 20px 40px 50px;
      font-size: 10px; letter-spacing: 2px; color: var(--text-dim);
      border-top: 1px solid rgba(255,255,255,0.04);
    }
    .footer span { color: var(--saffron); }

    /* Responsive */
    @media(max-width:700px){
      .header{padding:14px 20px;}
      .hero{padding:40px 20px 30px;}
      .hero-title{font-size:48px;letter-spacing:4px;}
      .stats-section,.actions-section{padding:10px 20px 30px;}
    }
  </style>
</head>
<body>
  <div class="tristrip"></div>
  <canvas id="bg-canvas"></canvas>
  <div class="scanlines"></div>

  <div class="page">
    <!-- HEADER -->
    <header class="header" id="hdr">
      <div class="brand-wrap">
        <span class="chakra">☸</span>
        <div>
          <div class="brand-name">ROAD<em>WATCH</em> AI</div>
          <div class="brand-sub">NATIONAL ROAD INFRASTRUCTURE MONITORING PORTAL · DIGITAL INDIA</div>
        </div>
      </div>
      <div class="header-right">
        <div class="live-badge"><div class="pulse-dot"></div>SYSTEM ONLINE</div>
        <div class="clock" id="clock">--:--:--</div>
      </div>
    </header>

    <!-- HERO -->
    <section class="hero">
      <div class="hero-eyebrow" id="eyebrow">▸ BHILAI-DURG DISTRICT COMMAND CENTER</div>
      <h1 class="hero-title" id="htitle">
        ROAD<span class="accent">WATCH</span><br>INTELLIGENCE
      </h1>
      <p class="hero-sub" id="hsub">
        Real-time pothole detection &amp; infrastructure monitoring across
        Bhilai, Durg, and surrounding zones — powered by YOLO v11.
      </p>
      <div class="glow-line" id="gline"></div>
    </section>

    <!-- LIVE STATS -->
    <section class="stats-section">
      <div class="stats-label">◈ LIVE INFRASTRUCTURE METRICS</div>
      <div class="stats-grid" id="stats-grid">
        <div class="stat-card" style="--accent:#FF9933" data-key="total" data-icon="⬡">
          <div class="stat-icon">⬡</div>
          <div class="stat-label">DETECTED</div>
          <div class="stat-value">—</div>
        </div>
        <div class="stat-card" style="--accent:#FF3D57" data-key="pending" data-icon="⚠">
          <div class="stat-icon">⚠</div>
          <div class="stat-label">PENDING</div>
          <div class="stat-value">—</div>
        </div>
        <div class="stat-card" style="--accent:#00B4FF" data-key="in_progress" data-icon="⟳">
          <div class="stat-icon">⟳</div>
          <div class="stat-label">IN PROGRESS</div>
          <div class="stat-value">—</div>
        </div>
        <div class="stat-card" style="--accent:#00E676" data-key="fixed" data-icon="✓">
          <div class="stat-icon">✓</div>
          <div class="stat-label">FIXED</div>
          <div class="stat-value">—</div>
        </div>
        <div class="stat-card" style="--accent:#FF9933" data-key="high_severity" data-icon="▲">
          <div class="stat-icon">▲</div>
          <div class="stat-label">HIGH SEVERITY</div>
          <div class="stat-value">—</div>
        </div>
        <div class="stat-card" style="--accent:#00E676" data-key="fix_rate" data-icon="◎" data-unit="%">
          <div class="stat-icon">◎</div>
          <div class="stat-label">FIX RATE</div>
          <div class="stat-value">—<span class="stat-unit">%</span></div>
        </div>
      </div>
    </section>

    <!-- ACTION CARDS -->
    <section class="actions-section">
      <div class="actions-grid" id="actions-grid">
        <a class="action-card" href="/camera/start" style="--card-glow:255,61,87">
          <div class="card-bg"></div>
          <div class="card-icon-wrap" style="--card-glow:255,61,87">🎥</div>
          <div class="card-title">LIVE SURVEILLANCE</div>
          <div class="card-desc">
            Real-time camera feed with YOLO v11 pothole detection,
            bounding-box overlay, and automated incident logging.
          </div>
          <div class="card-arrow"><div class="arrow-line"></div>ENTER FEED</div>
        </a>
        <a class="action-card" href="/dashboard/" style="--card-glow:0,180,255">
          <div class="card-bg"></div>
          <div class="card-icon-wrap" style="--card-glow:0,180,255">📊</div>
          <div class="card-title">ANALYTICS DASHBOARD</div>
          <div class="card-desc">
            Full Plotly Dash portal — hourly trends, zone heat maps,
            repair funnels, and the live incident log.
          </div>
          <div class="card-arrow"><div class="arrow-line"></div>OPEN PORTAL</div>
        </a>
        <a class="action-card" href="/api/health" target="_blank" style="--card-glow:0,230,118">
          <div class="card-bg"></div>
          <div class="card-icon-wrap" style="--card-glow:0,230,118">🛡</div>
          <div class="card-title">SYSTEM HEALTH</div>
          <div class="card-desc">
            Check MongoDB connection status, YOLO model readiness,
            and live API health endpoint output.
          </div>
          <div class="card-arrow"><div class="arrow-line"></div>VIEW STATUS</div>
        </a>
      </div>
    </section>

    <footer class="footer">
      © ROADWATCH AI &nbsp;·&nbsp; BHILAI-DURG, CHHATTISGARH &nbsp;·&nbsp;
      <span>DIGITAL INDIA INITIATIVE</span> &nbsp;·&nbsp;
      YOLO v11 · OPENCV · MONGODB · PLOTLY
    </footer>
  </div>

  <!-- HOTSPOT TICKER -->
  <div class="ticker-wrap">
    <div class="ticker-label">⚠ HOTSPOTS</div>
    <div style="overflow:hidden;flex:1">
      <div class="ticker-track" id="ticker">
        <span class="ticker-item">
          <span class="zone">BHILAI</span><span class="cnt">—</span>
        </span>
        <span class="ticker-sep">◆</span>
        <span class="ticker-item">
          <span class="zone">DURG</span><span class="cnt">—</span>
        </span>
        <span class="ticker-sep">◆</span>
        <span class="ticker-item">
          <span class="zone">LOADING HOTSPOT DATA...</span>
        </span>
      </div>
    </div>
  </div>

  <script>
  /* ══════════════════════════════════════════════════════
   * THREE.JS — Infinite 3D Road Scene
   * ══════════════════════════════════════════════════════ */
  (function() {
    const canvas = document.getElementById('bg-canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x0A0F1E, 1);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0A0F1E, 0.045);

    const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 200);
    camera.position.set(0, 3.5, 0);
    camera.lookAt(0, 1.5, -20);

    /* ── Road plane ── */
    const roadGeo = new THREE.PlaneGeometry(14, 200, 1, 1);
    const roadMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const road = new THREE.Mesh(roadGeo, roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.z = -80;
    scene.add(road);

    /* ── Ground plane ── */
    const groundGeo = new THREE.PlaneGeometry(400, 400);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x080C18, roughness: 1 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    scene.add(ground);

    /* ── Road lane markings (dashes) ── */
    for (let z = -2; z > -180; z -= 8) {
      const dashGeo = new THREE.PlaneGeometry(0.25, 3.5);
      const dashMat = new THREE.MeshStandardMaterial({
        color: 0xFFFFFF, roughness: 0.5, emissive: 0xFFFFFF, emissiveIntensity: 0.15,
      });
      const dash = new THREE.Mesh(dashGeo, dashMat);
      dash.rotation.x = -Math.PI / 2;
      dash.position.set(0, 0.01, z);
      scene.add(dash);
    }

    /* ── Pothole craters ── */
    const craters = [];
    const potholePositions = [
      [-3, -8], [4, -20], [-1, -35], [3.5, -50],
      [-4, -65], [2, -80], [-2.5, -92], [4.5, -105],
      [0, -120], [-3.5, -135], [3, -150],
    ];
    potholePositions.forEach(([x, z]) => {
      const g = new THREE.CircleGeometry(0.6 + Math.random() * 0.5, 16);
      const m = new THREE.MeshStandardMaterial({
        color: 0x050912, roughness: 1,
        emissive: 0xFF6600, emissiveIntensity: 0,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(x, 0.02, z);
      scene.add(mesh);

      /* Glow ring */
      const rg = new THREE.RingGeometry(0.65, 0.9, 24);
      const rm = new THREE.MeshBasicMaterial({
        color: 0xFF6600, transparent: true, opacity: 0, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(rg, rm);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.03, z);
      scene.add(ring);

      craters.push({ mesh, ring, phase: Math.random() * Math.PI * 2 });
    });

    /* ── Grid side lines ── */
    for (let i = 0; i < 14; i++) {
      const x = (i - 7) * 3;
      const pts = [new THREE.Vector3(x, 0, -200), new THREE.Vector3(x, 0, 10)];
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      const m = new THREE.LineBasicMaterial({
        color: i % 7 === 0 ? 0xFF9933 : 0x1a2a44,
        transparent: true, opacity: i % 7 === 0 ? 0.4 : 0.15,
      });
      scene.add(new THREE.Line(g, m));
    }

    /* ── Floating particles ── */
    const partGeo = new THREE.BufferGeometry();
    const partCount = 300;
    const positions = new Float32Array(partCount * 3);
    for (let i = 0; i < partCount; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * 60;
      positions[i * 3 + 1] = Math.random() * 15;
      positions[i * 3 + 2] = -Math.random() * 180;
    }
    partGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const partMat = new THREE.PointsMaterial({
      color: 0xFF9933, size: 0.08, transparent: true, opacity: 0.5,
    });
    scene.add(new THREE.Points(partGeo, partMat));

    /* ── Lights ── */
    scene.add(new THREE.AmbientLight(0xffffff, 0.15));
    const saffronLight = new THREE.PointLight(0xFF9933, 2, 30);
    saffronLight.position.set(0, 8, -10);
    scene.add(saffronLight);
    const greenLight = new THREE.PointLight(0x00E676, 1.5, 25);
    greenLight.position.set(-6, 5, -30);
    scene.add(greenLight);

    /* ── Roadside poles ── */
    [-7, 7].forEach(x => {
      for (let z = -10; z > -160; z -= 20) {
        const pole = new THREE.Mesh(
          new THREE.CylinderGeometry(0.06, 0.06, 5, 8),
          new THREE.MeshStandardMaterial({ color: 0x1a2a44, roughness: 0.8 }),
        );
        pole.position.set(x, 2.5, z);
        scene.add(pole);

        /* Lamp glow */
        const lamp = new THREE.PointLight(x > 0 ? 0xFF9933 : 0x00E676, 0.8, 10);
        lamp.position.set(x, 5.2, z);
        scene.add(lamp);

        const head = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 8, 8),
          new THREE.MeshStandardMaterial({
            emissive: x > 0 ? 0xFF9933 : 0x00E676,
            emissiveIntensity: 2, roughness: 0,
          }),
        );
        head.position.set(x, 5.3, z);
        scene.add(head);
      }
    });

    /* ── Animate ── */
    let t = 0;
    const roadOffset = { z: 0 };
    function animate() {
      requestAnimationFrame(animate);
      t += 0.008;

      /* Road scroll illusion */
      roadOffset.z = (roadOffset.z + 0.12) % 8;
      road.position.z = -80 + (roadOffset.z % 200);

      /* Camera gentle sway */
      camera.position.x = Math.sin(t * 0.3) * 0.4;
      camera.lookAt(Math.sin(t * 0.3) * 0.2, 1.5, -20);

      /* Pothole pulse */
      craters.forEach(c => {
        const glow = (Math.sin(t * 2 + c.phase) + 1) / 2;
        c.mesh.material.emissiveIntensity = glow * 0.4;
        c.ring.material.opacity = glow * 0.6;
        c.ring.scale.setScalar(1 + glow * 0.15);
      });

      /* Saffron light float */
      saffronLight.intensity = 1.5 + Math.sin(t * 1.5) * 0.5;
      greenLight.intensity = 1 + Math.sin(t * 1.2 + 1) * 0.5;

      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  })();


  /* ══════════════════════════════════════════════════════
   * FRAMER MOTION — Entrance Animations
   * ══════════════════════════════════════════════════════ */
  window.addEventListener('DOMContentLoaded', () => {
    const { animate, stagger } = window.Motion;

    /* Header slide down */
    animate('#hdr', { opacity: [0, 1], y: [-30, 0] }, { duration: 0.7, easing: [0.22, 1, 0.36, 1] });

    /* Hero sequence */
    animate('#eyebrow', { opacity: [0, 1], y: [20, 0] }, { duration: 0.6, delay: 0.3, easing: 'ease-out' });
    animate('#htitle',  { opacity: [0, 1], y: [40, 0] }, { duration: 0.8, delay: 0.5, easing: [0.22, 1, 0.36, 1] });
    animate('#hsub',    { opacity: [0, 1], y: [20, 0] }, { duration: 0.6, delay: 0.8, easing: 'ease-out' });
    animate('#gline',   { opacity: [0, 1],scaleX:[0,1] },{ duration: 0.8, delay: 1.0, easing: [0.22,1,0.36,1] });

    /* Stat cards stagger */
    animate('.stat-card',
      { opacity: [0, 1], y: [50, 0], scale: [0.92, 1] },
      { delay: stagger(0.1, { start: 1.1 }), duration: 0.65, easing: [0.22, 1, 0.36, 1] }
    );

    /* Action cards stagger */
    animate('.action-card',
      { opacity: [0, 1], x: [-40, 0] },
      { delay: stagger(0.12, { start: 1.6 }), duration: 0.65, easing: [0.22, 1, 0.36, 1] }
    );
  });


  /* ══════════════════════════════════════════════════════
   * LIVE STATS — Fetch & animated counter
   * ══════════════════════════════════════════════════════ */
  function animateCount(el, from, to, unit) {
    const isFloat = String(to).includes('.');
    const duration = 900;
    const start = performance.now();
    const run = (now) => {
      const p = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      const val = from + (to - from) * ease;
      el.innerHTML = isFloat
        ? val.toFixed(2) + '<span class="stat-unit">' + (unit||'') + '</span>'
        : Math.round(val).toLocaleString() + (unit && !isFloat ? '<span class="stat-unit">' + unit + '</span>' : '');
      if (p < 1) requestAnimationFrame(run);
    };
    requestAnimationFrame(run);
  }

  const prevVals = {};

  async function fetchStats() {
    try {
      const r = await fetch('/api/stats');
      const d = await r.json();
      document.querySelectorAll('.stat-card').forEach(card => {
        const key  = card.dataset.key;
        const unit = card.dataset.unit || '';
        const val  = key === 'fix_rate' ? d.fix_rate : (d[key] ?? 0);
        const numVal = parseFloat(val);
        const valueEl = card.querySelector('.stat-value');
        const prev = prevVals[key] ?? 0;
        if (prev !== numVal) {
          animateCount(valueEl, prev, numVal, unit);
          prevVals[key] = numVal;
        }
      });
    } catch(e) { console.warn('Stats fetch failed', e); }
  }

  async function fetchHotspots() {
    try {
      const r = await fetch('/api/hotspots');
      const spots = await r.json();
      if (!spots.length) return;
      const items = spots.map(s =>
        `<span class="ticker-item"><span class="zone">${s.zone}</span><span class="cnt">${s.count}</span></span><span class="ticker-sep">◆</span>`
      ).join('');
      document.getElementById('ticker').innerHTML = items + items; /* duplicate for seamless loop */
    } catch(e) {}
  }

  /* Clock */
  !function tick() {
    const d = new Date();
    document.getElementById('clock').textContent =
      d.toLocaleTimeString('en-IN', { hour12: false });
    setTimeout(tick, 1000);
  }();

  fetchStats();
  fetchHotspots();
  setInterval(fetchStats, 4000);

  /* ── 3D card tilt on mouse ── */
  document.querySelectorAll('.stat-card, .action-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width  - 0.5) * 14;
      const y = ((e.clientY - r.top)  / r.height - 0.5) * -14;
      card.style.transform = `perspective(800px) rotateX(${y}deg) rotateY(${x}deg) translateY(-6px) scale(1.02)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
  </script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════
# CAMERA PAGE — Enhanced 3D Surveillance Interface
# ══════════════════════════════════════════════════════════════════════
CAMERA_PAGE_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ title }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/framer-motion@11/dist/framer-motion.js"></script>
  <style>
    :root {
      --saffron:    #FF9933;
      --green:      #00E676;
      --red:        #FF3D57;
      --blue:       #00B4FF;
      --navy:       #0A0F1E;
      --navy2:      #0D1627;
      --text:       #C8D6E5;
      --text-dim:   #5A7A9A;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: var(--navy); }
    body {
      display: flex; flex-direction: column;
      font-family: 'JetBrains Mono', monospace;
      -webkit-font-smoothing: antialiased;
    }

    .tristrip {
      position: fixed; top: 0; left: 0; right: 0; height: 4px; z-index: 9999;
      background: linear-gradient(90deg, #FF9933 0% 33%, #fff 33% 66%, #138808 66% 100%);
    }

    /* ── Header ── */
    .header {
      flex-shrink: 0; display: flex; justify-content: space-between; align-items: center;
      padding: 12px 28px; margin-top: 4px;
      background: var(--navy2);
      border-bottom: 2px solid rgba(255,153,51,0.25);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .chakra { font-size: 26px; color: var(--saffron); animation: spin3d 10s linear infinite; display: inline-block; }
    @keyframes spin3d { to { transform: rotateY(360deg); } }
    .brand-name {
      font-family: 'Orbitron', sans-serif;
      font-size: 17px; font-weight: 900; letter-spacing: 3px; color: #fff;
      text-shadow: 0 0 15px rgba(255,153,51,0.4);
    }
    .brand-name em { color: var(--saffron); font-style: normal; }
    .brand-sub { font-size: 8.5px; letter-spacing: 2.5px; color: var(--text-dim); margin-top: 2px; }

    .header-right { display: flex; align-items: center; gap: 16px; }
    .btn-end {
      font-family: 'Orbitron', sans-serif;
      font-size: 10px; font-weight: 700; letter-spacing: 2px;
      color: var(--red); text-decoration: none;
      padding: 7px 16px; border-radius: 4px;
      border: 1px solid rgba(255,61,87,0.35);
      background: rgba(255,61,87,0.06);
      transition: all 0.2s;
    }
    .btn-end:hover { background: rgba(255,61,87,0.15); border-color: var(--red); }

    .cam-sel {
      padding: 7px 14px; border-radius: 5px;
      border: 1px solid rgba(255,153,51,0.2);
      background: rgba(13,22,39,0.9); color: var(--text);
      font-family: 'JetBrains Mono', monospace; font-size: 11px;
      cursor: pointer;
    }

    /* ── Stage ── */
    .stage {
      flex: 1; display: flex; gap: 0; overflow: hidden;
    }

    /* ── Camera Feed ── */
    .feed-wrap {
      flex: 1; position: relative; display: flex;
      align-items: center; justify-content: center;
      background: #000; overflow: hidden;
    }
    video { display: block; max-width: 100%; max-height: 100%; z-index: 1; }
    #overlayCanvas {
      position: absolute; top: 0; left: 0;
      width: 100%; height: 100%;
      z-index: 2; pointer-events: none;
    }

    /* ── Scan animation over video ── */
    .scanbar {
      position: absolute; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, var(--green), transparent);
      box-shadow: 0 0 12px var(--green);
      animation: scanv 3s ease-in-out infinite;
      z-index: 3; pointer-events: none;
    }
    @keyframes scanv {
      0%  { top: 0%;   opacity: 0.8; }
      50% { top: 100%; opacity: 0.8; }
      51% { top: 100%; opacity: 0; }
      52% { top: 0%;   opacity: 0; }
      100%{ top: 0%;   opacity: 0.8; }
    }

    /* ── Corner brackets ── */
    .corner { position: absolute; width: 22px; height: 22px; z-index: 4; }
    .corner-tl { top:12px; left:12px;  border-top:2px solid var(--saffron); border-left:2px solid var(--saffron); }
    .corner-tr { top:12px; right:12px; border-top:2px solid var(--saffron); border-right:2px solid var(--saffron); }
    .corner-bl { bottom:12px; left:12px;  border-bottom:2px solid var(--saffron); border-left:2px solid var(--saffron); }
    .corner-br { bottom:12px; right:12px; border-bottom:2px solid var(--saffron); border-right:2px solid var(--saffron); }

    /* ── Badges ── */
    .badge {
      position: absolute; z-index: 5;
      background: rgba(10,15,30,0.88);
      backdrop-filter: blur(10px);
      border-radius: 5px; padding: 7px 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px; font-weight: 500;
      letter-spacing: 1.5px; color: var(--text);
    }
    .badge-status { top: 14px; left: 14px; border-left: 2px solid var(--green); }
    .badge-mode   { top: 14px; right: 14px;
      font-family: 'Orbitron', sans-serif; font-size: 9px; font-weight: 700; letter-spacing: 2px;
      border-left: 2px solid var(--saffron); color: var(--saffron); }
    .badge-time   { bottom: 14px; right: 14px; color: var(--text-dim); font-size: 10px; }

    /* ── Alert banner (detection) ── */
    #alert-banner {
      position: absolute; left: 14px; right: 14px; bottom: 50px;
      background: rgba(255,61,87,0.12);
      border: 1px solid rgba(255,61,87,0.5);
      border-radius: 6px; padding: 10px 18px;
      display: flex; align-items: center; gap: 14px;
      z-index: 6; opacity: 0; transform: translateY(20px);
      backdrop-filter: blur(8px);
    }
    .alert-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--red); box-shadow: 0 0 10px var(--red);
      animation: blink 0.8s infinite;
    }
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
    .alert-text { font-family: 'Orbitron', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; color: var(--red); }
    .alert-meta { font-size: 10px; color: var(--text-dim); letter-spacing: 1px; margin-left: auto; }

    /* ── Side panel ── */
    .side-panel {
      width: 260px; flex-shrink: 0;
      background: var(--navy2);
      border-left: 1px solid rgba(255,153,51,0.12);
      display: flex; flex-direction: column;
      overflow-y: auto; padding: 20px 16px;
      gap: 18px;
    }
    .panel-section {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 8px; padding: 16px;
      position: relative; overflow: hidden;
    }
    .panel-section::before {
      content: ''; position: absolute;
      top: 0; left: 0; right: 0; height: 1px;
      background: var(--accent, var(--saffron));
      box-shadow: 0 0 8px var(--accent, var(--saffron));
    }
    .ps-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 8.5px; font-weight: 700; letter-spacing: 3px;
      color: var(--text-dim); margin-bottom: 14px;
    }

    /* Detection count big number */
    .det-count {
      font-family: 'Orbitron', sans-serif;
      font-size: 52px; font-weight: 900; text-align: center;
      color: #fff; line-height: 1;
      text-shadow: 0 0 30px rgba(255,153,51,0.5);
    }
    .det-label { text-align: center; font-size: 9px; letter-spacing: 3px; color: var(--text-dim); margin-top: 6px; }

    /* Mini stat rows */
    .mini-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 10px;
    }
    .mini-row:last-child { border-bottom: none; }
    .mini-key { color: var(--text-dim); letter-spacing: 1px; }
    .mini-val { font-family: 'Orbitron', sans-serif; font-size: 12px; font-weight: 700; }
    .mini-val.green { color: var(--green); }
    .mini-val.red   { color: var(--red);   }
    .mini-val.amber { color: #FFA000;      }
    .mini-val.blue  { color: var(--blue);  }

    /* Log feed */
    .log-feed { display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow-y: auto; }
    .log-entry {
      font-size: 10px; letter-spacing: 0.5px; line-height: 1.6;
      padding: 7px 10px; border-radius: 5px;
      background: rgba(255,255,255,0.03);
      border-left: 2px solid var(--green);
      animation: slideIn 0.4s ease-out;
    }
    @keyframes slideIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
    .log-entry .ltime { color: var(--text-dim); font-size: 9px; }
    .log-entry .ltype { color: var(--saffron); font-weight: 600; }
    .log-entry .lsev  { font-size: 9px; }
    .log-entry .lsev.high   { color: var(--red); }
    .log-entry .lsev.medium { color: #FFA000; }
    .log-entry .lsev.low    { color: var(--green); }

    /* Pulse dot */
    .pd { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
    .pd-green { background: var(--green); box-shadow: 0 0 8px var(--green); animation: blink 1.4s infinite; }
    .pd-red   { background: var(--red);   box-shadow: 0 0 8px var(--red);   animation: blink 0.8s infinite; }

    /* ── Footer ── */
    .footer {
      flex-shrink: 0; display: flex; justify-content: space-between; align-items: center;
      padding: 8px 28px;
      background: var(--navy2); border-top: 1px solid rgba(255,255,255,0.06);
      font-size: 9px; letter-spacing: 2px; color: var(--text-dim);
    }

    /* Particle canvas */
    #particle-canvas {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      z-index: 3; pointer-events: none;
    }

    @media(max-width:800px){ .side-panel{display:none;} }
  </style>
</head>
<body>
  <div class="tristrip"></div>

  <!-- HEADER -->
  <header class="header" id="hdr">
    <div class="brand">
      <span class="chakra">☸</span>
      <div>
        <div class="brand-name">ROAD<em>WATCH</em> AI</div>
        <div class="brand-sub">LIVE SURVEILLANCE MODULE · {{ subtitle }}</div>
      </div>
    </div>
    <div class="header-right">
      <select class="cam-sel" id="camera-select"></select>
      <a href="{{ dashboard_url }}" class="btn-end">✕ END SESSION</a>
    </div>
  </header>

  <!-- STAGE -->
  <div class="stage">
    <!-- VIDEO FEED -->
    <div class="feed-wrap" id="feed-wrap">
      <video id="videoElement" autoplay playsinline></video>
      <canvas id="overlayCanvas"></canvas>
      <canvas id="particle-canvas"></canvas>
      <div class="scanbar"></div>

      <!-- Corners -->
      <div class="corner corner-tl"></div>
      <div class="corner corner-tr"></div>
      <div class="corner corner-bl"></div>
      <div class="corner corner-br"></div>

      <!-- Badges -->
      <div class="badge badge-status" id="statusText">
        <span class="pd pd-green"></span>INITIALIZING...
      </div>
      <div class="badge badge-mode">{{ feed_label }}</div>
      <div class="badge badge-time" id="timeOverlay">--:--:--</div>

      <!-- Alert banner -->
      <div id="alert-banner">
        <div class="alert-dot"></div>
        <div class="alert-text" id="alert-type">POTHOLE DETECTED</div>
        <div class="alert-meta" id="alert-meta">CONF: 0.00</div>
      </div>
    </div>

    <!-- SIDE PANEL -->
    <div class="side-panel">
      <!-- Detection Count -->
      <div class="panel-section" style="--accent:var(--saffron)">
        <div class="ps-title">◈ SESSION DETECTIONS</div>
        <div class="det-count" id="det-count">0</div>
        <div class="det-label">INCIDENTS LOGGED</div>
      </div>

      <!-- Live Stats -->
      <div class="panel-section" style="--accent:var(--blue)">
        <div class="ps-title">◈ INFRASTRUCTURE STATUS</div>
        <div class="mini-row"><span class="mini-key">TOTAL</span><span class="mini-val" id="sp-total">—</span></div>
        <div class="mini-row"><span class="mini-key">PENDING</span><span class="mini-val red" id="sp-pending">—</span></div>
        <div class="mini-row"><span class="mini-key">IN PROGRESS</span><span class="mini-val amber" id="sp-progress">—</span></div>
        <div class="mini-row"><span class="mini-key">FIXED</span><span class="mini-val green" id="sp-fixed">—</span></div>
        <div class="mini-row"><span class="mini-key">FIX RATE</span><span class="mini-val blue" id="sp-rate">—</span></div>
      </div>

      <!-- Detection Log -->
      <div class="panel-section" style="--accent:var(--green); flex:1">
        <div class="ps-title">◈ DETECTION LOG</div>
        <div class="log-feed" id="log-feed">
          <div class="log-entry">
            <div class="ltime">AWAITING FEED...</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <footer class="footer">
    <span>ROADWATCH AI · BHILAI-DURG · CHHATTISGARH</span>
    <span id="footer-clock">--:--:--  --/--/----</span>
    <span>YOLO v11 · OPENCV · MONGODB</span>
  </footer>

  <script>
  /* ══════════════════════════════════════════════════════
   * FRAMER MOTION — Entrance
   * ══════════════════════════════════════════════════════ */
  window.addEventListener('DOMContentLoaded', () => {
    const { animate, stagger } = window.Motion;
    animate('#hdr', { opacity:[0,1], y:[-20,0] }, { duration:0.5, easing:[0.22,1,0.36,1] });
    animate('.panel-section', { opacity:[0,1], x:[30,0] }, {
      delay: stagger(0.12, {start:0.3}), duration:0.55, easing:[0.22,1,0.36,1]
    });
  });

  /* ══════════════════════════════════════════════════════
   * DOM REFS
   * ══════════════════════════════════════════════════════ */
  const video         = document.getElementById('videoElement');
  const overlay       = document.getElementById('overlayCanvas');
  const particleCvs   = document.getElementById('particle-canvas');
  const ctx           = overlay.getContext('2d');
  const pCtx          = particleCvs.getContext('2d');
  const capCanvas     = document.createElement('canvas');
  const capCtx        = capCanvas.getContext('2d');
  const camSel        = document.getElementById('camera-select');
  const statusEl      = document.getElementById('statusText');
  const timeEl        = document.getElementById('timeOverlay');
  const footerClock   = document.getElementById('footer-clock');
  const alertBanner   = document.getElementById('alert-banner');
  const alertType     = document.getElementById('alert-type');
  const alertMeta     = document.getElementById('alert-meta');
  const detCountEl    = document.getElementById('det-count');
  const logFeed       = document.getElementById('log-feed');

  let stream = null, loop = null;
  let sessionDetections = 0;
  let alertTimeout = null;
  const API  = '{{ detect_url }}';
  const RATE = {{ detect_interval }};

  /* ── Particle system ── */
  const particles = [];
  function spawnParticles(x, y, count) {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1 + Math.random() * 4;
      particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1, decay: 0.02 + Math.random() * 0.04,
        radius: 2 + Math.random() * 4,
        color: Math.random() > 0.5 ? '#FF9933' : '#FF3D57',
      });
    }
  }
  function tickParticles() {
    pCtx.clearRect(0, 0, particleCvs.width, particleCvs.height);
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy;
      p.vy += 0.1; // gravity
      p.life -= p.decay;
      if (p.life <= 0) { particles.splice(i, 1); continue; }
      pCtx.globalAlpha = p.life;
      pCtx.beginPath();
      pCtx.arc(p.x, p.y, p.radius * p.life, 0, Math.PI * 2);
      pCtx.fillStyle = p.color;
      pCtx.fill();
    }
    pCtx.globalAlpha = 1;
    requestAnimationFrame(tickParticles);
  }
  tickParticles();

  /* ── Clock ── */
  !function tick() {
    const d = new Date();
    const t = d.toLocaleTimeString('en-IN',{hour12:false});
    const dt = d.toLocaleDateString('en-IN');
    timeEl.textContent = t;
    footerClock.textContent = t + '  ' + dt;
    setTimeout(tick, 1000);
  }();

  /* ── Status helper ── */
  function setStatus(msg, isAlert) {
    const dot = isAlert
      ? '<span class="pd pd-red"></span>'
      : '<span class="pd pd-green"></span>';
    statusEl.innerHTML = dot + msg;
  }

  /* ── Show alert banner (Framer Motion) ── */
  function showAlert(type, conf) {
    const { animate } = window.Motion;
    alertType.textContent = 'DETECTED: ' + type.toUpperCase();
    alertMeta.textContent = 'CONF: ' + conf.toFixed(2);
    animate('#alert-banner', { opacity:[0,1], y:[20,0] }, { duration:0.35, easing:[0.22,1,0.36,1] });
    if (alertTimeout) clearTimeout(alertTimeout);
    alertTimeout = setTimeout(() => {
      animate('#alert-banner', { opacity:[1,0], y:[0,12] }, { duration:0.4, easing:'ease-in' });
    }, 2500);
  }

  /* ── Log entry ── */
  function addLog(type, conf, sev) {
    const d = new Date();
    const t = d.toLocaleTimeString('en-IN',{hour12:false});
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const sevClass = sev === 'High' ? 'high' : sev === 'Low' ? 'low' : 'medium';
    entry.innerHTML = `
      <div class="ltime">${t}</div>
      <div><span class="ltype">${type}</span> · <span class="lsev ${sevClass}">${sev.toUpperCase()}</span></div>
      <div class="ltime">CONF: ${conf.toFixed(2)}</div>`;
    if (logFeed.children.length === 1 && logFeed.children[0].querySelector('.ltime')?.textContent === 'AWAITING FEED...') {
      logFeed.innerHTML = '';
    }
    logFeed.prepend(entry);
    while (logFeed.children.length > 15) logFeed.removeChild(logFeed.lastChild);
  }

  /* ── Live stats ── */
  async function fetchStats() {
    try {
      const r = await fetch('/api/stats');
      const d = await r.json();
      const set = (id, v) => { const el = document.getElementById(id); if(el) el.textContent = v; };
      set('sp-total',    (d.total ?? 0).toLocaleString());
      set('sp-pending',  (d.pending ?? 0).toLocaleString());
      set('sp-progress', (d.in_progress ?? 0).toLocaleString());
      set('sp-fixed',    (d.fixed ?? 0).toLocaleString());
      set('sp-rate',     (d.fix_rate ?? 0) + '%');
    } catch(e) {}
  }
  fetchStats();
  setInterval(fetchStats, 5000);

  /* ── Camera init ── */
  async function init() {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      camSel.innerHTML = '';
      devs.forEach((d,i) => {
        const o = document.createElement('option');
        o.value = d.deviceId; o.text = d.label || 'Camera '+(i+1);
        camSel.appendChild(o);
      });
      devs.length ? startCam(devs[0].deviceId) : setStatus('NO CAMERAS FOUND', true);
    } catch(e) { setStatus('CAMERA ACCESS DENIED', true); }
  }

  async function startCam(id) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: id ? { deviceId: { exact: id } } : true });
      video.srcObject = stream;
      setStatus('LIVE STREAM ACTIVE', false);
      video.onloadedmetadata = () => {
        overlay.width  = capCanvas.width  = particleCvs.width  = video.videoWidth;
        overlay.height = capCanvas.height = particleCvs.height = video.videoHeight;
        if (loop) clearInterval(loop);
        loop = setInterval(detect, RATE);
      };
    } catch(e) { setStatus('STREAM FAILED', true); }
  }
  camSel.addEventListener('change', e => startCam(e.target.value));

  /* ── 3D bounding box draw ── */
  function drawBox3D(x1, y1, x2, y2, label) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const w = x2 - x1, h = y2 - y1;
    const depth = 18, d = depth;

    // Back face (dimmer)
    ctx.beginPath();
    ctx.moveTo(x1+d, y1-d); ctx.lineTo(x2+d, y1-d);
    ctx.lineTo(x2+d, y2-d); ctx.lineTo(x1+d, y2-d);
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255,153,51,0.25)'; ctx.lineWidth = 1.5;
    ctx.stroke();

    // Connecting edges
    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].forEach(([px,py]) => {
      ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+d,py-d);
      ctx.strokeStyle = 'rgba(255,153,51,0.25)'; ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Front face glow
    ctx.shadowColor = '#FF9933'; ctx.shadowBlur = 16;
    ctx.strokeStyle = '#FF9933'; ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, w, h);
    ctx.shadowBlur = 0;

    // Corner accents
    const cs = 14;
    const corners = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]];
    const dirs     = [[1,1],[-1,1],[-1,-1],[1,-1]];
    corners.forEach(([cx,cy],[dx,dy]) => {
      ctx.beginPath();
      ctx.moveTo(cx + dx*cs, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy*cs);
      ctx.strokeStyle = '#00E676'; ctx.lineWidth = 3;
      ctx.shadowColor = '#00E676'; ctx.shadowBlur = 10;
      ctx.stroke(); ctx.shadowBlur = 0;
    }, dirs);

    // Label tag
    const lbl = label;
    ctx.font = '700 13px "JetBrains Mono", monospace';
    const tw = ctx.measureText(lbl).width;
    const lx = x1, ly = Math.max(30, y1 - 10);
    ctx.fillStyle = 'rgba(10,15,30,0.88)';
    ctx.fillRect(lx, ly - 18, tw + 20, 22);
    ctx.strokeStyle = '#FF9933'; ctx.lineWidth = 1;
    ctx.strokeRect(lx, ly - 18, tw + 20, 22);
    ctx.fillStyle = '#FF9933'; ctx.shadowColor = '#FF9933'; ctx.shadowBlur = 6;
    ctx.fillText(lbl, lx + 10, ly - 1);
    ctx.shadowBlur = 0;
  }

  /* ── Detection loop ── */
  async function detect() {
    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;
    capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
    const img = capCanvas.toDataURL('image/jpeg', 0.8);
    try {
      const r = await (await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: img }),
      })).json();

      if (r.detected && r.bbox) {
        const [x1,y1,x2,y2] = r.bbox;
        const lbl = `${r.hazard_type} ${r.confidence.toFixed(2)}`;
        const sev = r.confidence > 0.75 ? 'High' : r.confidence > 0.5 ? 'Medium' : 'Low';

        drawBox3D(x1, y1, x2, y2, lbl);

        /* Particles at center of box */
        const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
        spawnParticles(cx, cy, 18);

        /* Animate detection count */
        sessionDetections++;
        const { animate } = window.Motion;
        animate(detCountEl,
          { scale:[1.3,1], color:['#FF9933','#ffffff'] },
          { duration:0.5, easing:[0.22,1,0.36,1] }
        );
        detCountEl.textContent = sessionDetections;

        showAlert(r.hazard_type, r.confidence);
        addLog(r.hazard_type, r.confidence, sev);
        setStatus('DETECTED: ' + r.hazard_type.toUpperCase(), true);
        setTimeout(() => { setStatus('LIVE STREAM ACTIVE', false); ctx.clearRect(0,0,overlay.width,overlay.height); }, 2000);

      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    } catch(e) { console.error('Detect error:', e); }
  }

  init();
  </script>
</body>
</html>
"""

CAMERA_COLORS = {
    "bg":           "#0A0F1E",
    "text":         "#C8D6E5",
    "saffron":      "#FF9933",
    "india_green":  "#00E676",
    "header_bg":    "#0D1627",
    "panel_border": "#1A2A44",
    "red":          "#FF3D57",
    "blue":         "#00B4FF",
}


def _camera_context() -> dict:
    return {
        "title":           "RoadWatch AI — Live Surveillance Session",
        "subtitle":        "BHILAI-DURG · CHHATTISGARH",
        "feed_label":      "SURVEILLANCE FEED",
        "dashboard_url":   "/dashboard/",
        "detect_url":      "/api/detect_frame",
        "detect_interval": 1500,
        "footer_left":     "ROADWATCH AI  ·  NATIONAL ROAD INFRASTRUCTURE PORTAL",
        "footer_right":    "POWERED BY YOLO v11 · OPENCV · MONGODB",
        "c":               CAMERA_COLORS,
    }


# ══════════════════════════════════════════════════════════════════════
# Flask Application Factory
# ══════════════════════════════════════════════════════════════════════


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = Config.FLASK_SECRET_KEY

    initialize_storage()
    seed_dummy_data()
    mount_dashboard(app)
    register_routes(app)

    print("RoadWatch AI Starting...")
    print(f"Model loaded: {get_model_status()}")
    print(f"MongoDB {'connected' if Config.DB_CONNECTED else 'fallback-active'}")
    print(f"Google Maps API {'ready' if Config.GOOGLE_MAPS_API_KEY else 'not configured'}")
    print(f"Home:      http://localhost:{Config.FLASK_PORT}/")
    print(f"Dashboard: http://localhost:{Config.FLASK_PORT}/dashboard/")
    print(f"Camera:    http://localhost:{Config.FLASK_PORT}/camera/start")
    print(f"API Base:  http://localhost:{Config.FLASK_PORT}/api/")
    return app


def register_routes(app: Flask) -> None:

    @app.post("/api/report")
    def api_report() -> Any:
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        lat        = float(payload.get("lat", 0.0))
        lng        = float(payload.get("lng", 0.0))
        image_path = str(payload.get("image_path", "")).strip()
        severity   = str(payload.get("severity", "Medium"))
        confidence = float(payload.get("confidence", 0.0))
        if not image_path:
            return jsonify({"error": "image_path is required"}), 400
        report = create_and_save_report(lat=lat, lng=lng, image_path=image_path,
                                        severity=severity, confidence=confidence)
        return jsonify({"status": "saved", "report": report}), 201

    @app.get("/api/potholes")
    def api_potholes() -> Any:
        limit = request.args.get("limit", default=50, type=int)
        potholes = get_all_potholes(limit=limit)
        return jsonify({"count": len(potholes), "potholes": potholes})

    @app.get("/api/stats")
    def api_stats() -> Any:
        counts   = get_counts()
        total    = counts.get("total", 0)
        fixed    = counts.get("fixed", 0)
        fix_rate = round((fixed / total) * 100, 2) if total else 0.0
        return jsonify({
            **counts,
            "fix_rate":        fix_rate,
            "hourly":          get_hourly_counts(hours=8),
            "zones":           get_zone_counts(),
            "status_counts":   get_status_counts(),
            "severity_counts": get_severity_counts(),
        })

    @app.post("/api/fix/<pothole_id>")
    def api_fix(pothole_id: str) -> Any:
        updated = mark_as_fixed(pothole_id)
        if not updated:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": "updated"})

    @app.get("/api/hotspots")
    def api_hotspots() -> Any:
        zones    = get_zone_counts()
        hotspots = [z for z in zones if z.get("count", 0) > 1]
        return jsonify(hotspots)

    @app.get("/api/health")
    def api_health() -> Any:
        return jsonify({
            "status": "ok",
            "db":     "connected" if Config.DB_CONNECTED else "fallback",
            "model":  get_model_status(),
        })

    @app.post("/api/detect_frame")
    def api_detect_frame() -> Any:
        payload   = request.get_json(silent=True) or {}
        image_b64 = payload.get("image", "")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400
        try:
            _, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
            data    = base64.b64decode(encoded)
            np_arr  = np.frombuffer(data, np.uint8)
            frame   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            result  = detect_frame(frame)
            if result.get("detected"):
                report = create_and_save_report(
                    lat=21.2350, lng=81.3462,
                    image_path=result["image_path"],
                    severity=result.get("severity", "Medium"),
                    confidence=result.get("confidence", 0.0),
                )
                return jsonify({
                    "detected":    True,
                    "hazard_type": report["hazard_type"],
                    "confidence":  result["confidence"],
                    "bbox":        result["bbox"],
                })
            return jsonify({"detected": False})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/camera/start")
    def start_camera() -> Any:
        return render_template_string(CAMERA_PAGE_TEMPLATE, **_camera_context())

    @app.get("/")
    def index() -> Any:
        return render_template_string(HOME_PAGE_TEMPLATE)


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", Config.FLASK_PORT if hasattr(Config, "FLASK_PORT") else 5000))
    app.run(host="0.0.0.0", port=port, debug=False)