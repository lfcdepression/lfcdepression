---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

教育经历
======
* 武汉大学，2022-至今，物理学学士

研究方向
======
- AI for Science（使用AI方法研究生物复杂系统，量子多体系统等有大量数据，且传统方法在建模上存在一定困难的方向）
- Science for AI（使用统计物理研究神经网络）
- 凝聚态计算
- 量子多体物理计算方法

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams
