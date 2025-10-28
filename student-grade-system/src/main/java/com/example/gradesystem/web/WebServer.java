package com.example.gradesystem.web;

import com.example.gradesystem.model.Course;
import com.example.gradesystem.model.Enrollment;
import com.example.gradesystem.model.Student;
import com.example.gradesystem.repo.CourseRepository;
import com.example.gradesystem.repo.EnrollmentRepository;
import com.example.gradesystem.repo.StudentRepository;
import com.example.gradesystem.service.GradeService;
import com.example.gradesystem.util.HttpUtil;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class WebServer {
    private HttpServer server;
    private GradeService gradeService;
    private StudentRepository studentRepo;
    private CourseRepository courseRepo;
    private EnrollmentRepository enrollmentRepo;

    public void start(int port) {
        try {
            Path dataDir = Path.of("data");
            this.studentRepo = new StudentRepository(dataDir.resolve("students.tsv"));
            this.courseRepo = new CourseRepository(dataDir.resolve("courses.tsv"));
            this.enrollmentRepo = new EnrollmentRepository(dataDir.resolve("enrollments.tsv"));
            this.gradeService = new GradeService(studentRepo, courseRepo, enrollmentRepo);

            this.server = HttpServer.create(new InetSocketAddress(port), 0);
            server.createContext("/api", new ApiHandler());
            server.createContext("/", new StaticHandler());
            server.setExecutor(null);
            server.start();
            System.out.println("Web server started at http://localhost:" + port);
            // Block until interrupted
            try {
                Thread.currentThread().join();
            } catch (InterruptedException ignored) {}
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private class ApiHandler implements HttpHandler {
        @Override public void handle(HttpExchange ex) throws IOException {
            try {
                String method = ex.getRequestMethod();
                URI uri = ex.getRequestURI();
                String path = uri.getPath().replaceFirst("^/api", "");
                Map<String,String> q = HttpUtil.parseQuery(uri.getQuery());

                if (path.equals("/students") && method.equals("GET")) {
                    String json = listStudentsJson();
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }
                if (path.equals("/students") && method.equals("POST")) {
                    Map<String,String> body = HttpUtil.readFormBody(ex);
                    String name = body.getOrDefault("name", "").trim();
                    if (name.isEmpty()) { HttpUtil.sendJson(ex, 400, "{\"error\":\"name required\"}"); return; }
                    Student s = new Student(UUID.randomUUID().toString(), name);
                    studentRepo.save(s);
                    HttpUtil.sendJson(ex, 200, studentToJson(s));
                    return;
                }
                if (path.startsWith("/students/") && method.equals("DELETE")) {
                    String id = path.substring("/students/".length());
                    boolean ok = studentRepo.deleteById(id);
                    HttpUtil.sendJson(ex, ok ? 200 : 404, "{\"ok\":" + ok + "}");
                    return;
                }
                if (path.startsWith("/students/") && path.endsWith("/gpa") && method.equals("GET")) {
                    String id = path.substring("/students/".length(), path.length() - "/gpa".length());
                    String term = q.getOrDefault("term", "");
                    var g = (term.isEmpty()) ? gradeService.getStudentGradesAndGpa(id) : gradeService.getStudentGradesAndGpa(id, term);
                    if (g == null) { HttpUtil.sendJson(ex, 404, "{\"error\":\"not found\"}"); return; }
                    String gradesJson = g.grades().entrySet().stream()
                        .map(e -> "{\"course\": " + courseToJson(e.getKey()) + ", \"grade\": " + (e.getValue()==null?"null":e.getValue().toString()) + "}")
                        .collect(Collectors.joining(","));
                    String json = "{\"student\": " + studentToJson(g.student()) + ", \"gpa\": " + String.format(Locale.US, "%.2f", g.gpa()) + ", \"grades\": [" + gradesJson + "]}";
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }

                if (path.equals("/courses") && method.equals("GET")) {
                    HttpUtil.sendJson(ex, 200, listCoursesJson());
                    return;
                }
                if (path.equals("/courses") && method.equals("POST")) {
                    Map<String,String> body = HttpUtil.readFormBody(ex);
                    String name = body.getOrDefault("name", "").trim();
                    int credits;
                    try { credits = Integer.parseInt(body.getOrDefault("credits", "0")); } catch (Exception e) { credits = 0; }
                    if (name.isEmpty() || credits <= 0) { HttpUtil.sendJson(ex, 400, "{\"error\":\"invalid\"}"); return; }
                    Course c = new Course(UUID.randomUUID().toString(), name, credits);
                    courseRepo.save(c);
                    HttpUtil.sendJson(ex, 200, courseToJson(c));
                    return;
                }
                if (path.startsWith("/courses/") && method.equals("DELETE")) {
                    String id = path.substring("/courses/".length());
                    boolean ok = courseRepo.deleteById(id);
                    HttpUtil.sendJson(ex, ok ? 200 : 404, "{\"ok\":" + ok + "}");
                    return;
                }
                if (path.startsWith("/courses/") && path.endsWith("/stats") && method.equals("GET")) {
                    String id = path.substring("/courses/".length(), path.length() - "/stats".length());
                    String term = q.getOrDefault("term", "");
                    var stats = (term.isEmpty()) ? gradeService.getCourseStats(id) : gradeService.getCourseStats(id, term);
                    if (stats == null) { HttpUtil.sendJson(ex, 404, "{\"error\":\"not found\"}"); return; }
                    String json = "{\"course\": " + courseToJson(stats.course()) + ", \"count\": " + stats.count() + ", \"average\": " + fmt(stats.average()) + ", \"max\": " + fmt(stats.max()) + ", \"min\": " + fmt(stats.min()) + "}";
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }

                if (path.equals("/enroll") && method.equals("POST")) {
                    Map<String,String> body = HttpUtil.readFormBody(ex);
                    String sid = body.getOrDefault("studentId", "");
                    String cid = body.getOrDefault("courseId", "");
                    String term = body.getOrDefault("term", "");
                    boolean ok = gradeService.enrollStudentInCourse(sid, cid, term);
                    HttpUtil.sendJson(ex, ok ? 200 : 400, "{\"ok\":" + ok + "}");
                    return;
                }
                if (path.equals("/grade") && method.equals("POST")) {
                    Map<String,String> body = HttpUtil.readFormBody(ex);
                    String sid = body.getOrDefault("studentId", "");
                    String cid = body.getOrDefault("courseId", "");
                    String term = body.getOrDefault("term", "");
                    double grade = 0;
                    try { grade = Double.parseDouble(body.getOrDefault("grade", "0")); } catch (Exception ignored) {}
                    boolean ok = (term.isEmpty()) ? gradeService.putGrade(sid, cid, grade) : gradeService.putGrade(sid, cid, term, grade);
                    HttpUtil.sendJson(ex, ok ? 200 : 400, "{\"ok\":" + ok + "}");
                    return;
                }

                if (path.equals("/analytics/leaderboard") && method.equals("GET")) {
                    int limit = 10;
                    try { limit = Integer.parseInt(q.getOrDefault("limit", "10")); } catch (Exception ignored) {}
                    String term = q.getOrDefault("term", "");
                    List<GradeService.StudentGpa> list = gradeService.leaderboard(limit, term);
                    String json = list.stream().map(sg -> "{\"student\": " + studentToJson(sg.student()) + ", \"gpa\": " + fmt(sg.gpa()) + "}").collect(Collectors.joining(",","[","]"));
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }
                if (path.equals("/analytics/course-difficulty") && method.equals("GET")) {
                    String term = q.getOrDefault("term", "");
                    Map<Course, Double> map = gradeService.courseDifficulty(term);
                    String json = map.entrySet().stream().map(e -> "{\"course\": " + courseToJson(e.getKey()) + ", \"average\": " + fmt(e.getValue()) + "}").collect(Collectors.joining(",","[","]"));
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }
                if (path.equals("/analytics/grade-distribution") && method.equals("GET")) {
                    String courseId = q.getOrDefault("courseId", "");
                    String term = q.getOrDefault("term", "");
                    int bins = 10; try { bins = Integer.parseInt(q.getOrDefault("bins", "10")); } catch (Exception ignored) {}
                    int[] hist = gradeService.gradeDistributionForCourse(courseId, term, bins);
                    String json = Arrays.stream(hist).mapToObj(String::valueOf).collect(Collectors.joining(",","[","]"));
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }

                if (path.equals("/search/students") && method.equals("GET")) {
                    String query = q.getOrDefault("q", "");
                    String json = gradeService.searchStudents(query).stream().map(WebServer::studentToJson).collect(Collectors.joining(",","[","]"));
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }
                if (path.equals("/search/courses") && method.equals("GET")) {
                    String query = q.getOrDefault("q", "");
                    String json = gradeService.searchCourses(query).stream().map(WebServer::courseToJson).collect(Collectors.joining(",","[","]"));
                    HttpUtil.sendJson(ex, 200, json);
                    return;
                }

                if (path.equals("/export/students.csv") && method.equals("GET")) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("id,name\n");
                    for (Student s : studentRepo.findAll()) sb.append(s.getId()).append(',').append(csv(s.getName())).append('\n');
                    HttpUtil.sendText(ex, 200, sb.toString(), "text/csv; charset=utf-8");
                    return;
                }
                if (path.equals("/export/courses.csv") && method.equals("GET")) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("id,name,credits\n");
                    for (Course c : courseRepo.findAll()) sb.append(c.getId()).append(',').append(csv(c.getName())).append(',').append(c.getCredits()).append('\n');
                    HttpUtil.sendText(ex, 200, sb.toString(), "text/csv; charset=utf-8");
                    return;
                }
                if (path.equals("/export/enrollments.csv") && method.equals("GET")) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("studentId,courseId,grade,term\n");
                    for (Enrollment e : enrollmentRepo.findAll()) sb.append(e.getStudentId()).append(',').append(e.getCourseId()).append(',').append(e.getGrade()==null?"":e.getGrade()).append(',').append(csv(e.getTerm())).append('\n');
                    HttpUtil.sendText(ex, 200, sb.toString(), "text/csv; charset=utf-8");
                    return;
                }

                HttpUtil.sendJson(ex, 404, "{\"error\":\"unknown endpoint\"}");
            } catch (Exception e) {
                e.printStackTrace();
                HttpUtil.sendJson(ex, 500, "{\"error\":\"" + HttpUtil.jsonEscape(e.getMessage()) + "\"}");
            }
        }
    }

    private class StaticHandler implements HttpHandler {
        @Override public void handle(HttpExchange ex) throws IOException {
            URI uri = ex.getRequestURI();
            String path = uri.getPath();
            if (path.equals("/")) path = "/index.html";
            Path file = Path.of("public" + path);
            if (!Files.exists(file) || Files.isDirectory(file)) {
                HttpUtil.sendText(ex, 404, "Not Found", "text/plain; charset=utf-8");
                return;
            }
            byte[] bytes = Files.readAllBytes(file);
            ex.getResponseHeaders().set("Content-Type", HttpUtil.contentTypeFor(file.getFileName().toString()));
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
        }
    }

    private static String fmt(double d) {
        return String.format(Locale.US, "%.2f", d);
    }

    private static String csv(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return '"' + s.replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static String studentToJson(Student s) {
        return "{\"id\":" + HttpUtil.jsonEscape(s.getId()) + ",\"name\":" + HttpUtil.jsonEscape(s.getName()) + "}";
    }

    private static String courseToJson(Course c) {
        return "{\"id\":" + HttpUtil.jsonEscape(c.getId()) + ",\"name\":" + HttpUtil.jsonEscape(c.getName()) + ",\"credits\":" + c.getCredits() + "}";
    }

    private static String listStudentsJson() {
        // Simple list without pagination for now
        Path dataDir = Path.of("data");
        StudentRepository repo = new StudentRepository(dataDir.resolve("students.tsv"));
        return repo.findAll().stream().map(WebServer::studentToJson).collect(Collectors.joining(",","[","]"));
    }

    private static String listCoursesJson() {
        Path dataDir = Path.of("data");
        CourseRepository repo = new CourseRepository(dataDir.resolve("courses.tsv"));
        return repo.findAll().stream().map(WebServer::courseToJson).collect(Collectors.joining(",","[","]"));
    }
}
