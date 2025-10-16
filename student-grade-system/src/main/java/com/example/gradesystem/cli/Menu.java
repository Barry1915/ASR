package com.example.gradesystem.cli;

import com.example.gradesystem.model.Course;
import com.example.gradesystem.model.Student;
import com.example.gradesystem.service.GradeService;
import com.example.gradesystem.repo.CourseRepository;
import com.example.gradesystem.repo.StudentRepository;
import com.example.gradesystem.repo.EnrollmentRepository;

import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;
import java.util.UUID;

public class Menu {
    private final Scanner scanner = new Scanner(System.in);
    private final StudentRepository studentRepo;
    private final CourseRepository courseRepo;
    private final EnrollmentRepository enrollmentRepo;
    private final GradeService gradeService;

    public Menu() {
        Path dataDir = Path.of("data");
        this.studentRepo = new StudentRepository(dataDir.resolve("students.tsv"));
        this.courseRepo = new CourseRepository(dataDir.resolve("courses.tsv"));
        this.enrollmentRepo = new EnrollmentRepository(dataDir.resolve("enrollments.tsv"));
        this.gradeService = new GradeService(studentRepo, courseRepo, enrollmentRepo);
    }

    public void start() {
        while (true) {
            System.out.println("\n=== 学生成绩系统 ===");
            System.out.println("1) 学生管理");
            System.out.println("2) 课程管理");
            System.out.println("3) 选课与成绩");
            System.out.println("0) 退出");
            System.out.print("选择: ");
            String choice = scanner.nextLine().trim();
            switch (choice) {
                case "1": studentsMenu(); break;
                case "2": coursesMenu(); break;
                case "3": enrollmentMenu(); break;
                case "0": System.out.println("再见!"); return;
                default: System.out.println("无效选择");
            }
        }
    }

    private void studentsMenu() {
        while (true) {
            System.out.println("\n-- 学生管理 --");
            System.out.println("1) 添加学生");
            System.out.println("2) 列出学生");
            System.out.println("3) 删除学生");
            System.out.println("0) 返回");
            System.out.print("选择: ");
            String c = scanner.nextLine().trim();
            switch (c) {
                case "1": addStudent(); break;
                case "2": listStudents(); break;
                case "3": deleteStudent(); break;
                case "0": return;
                default: System.out.println("无效选择");
            }
        }
    }

    private void coursesMenu() {
        while (true) {
            System.out.println("\n-- 课程管理 --");
            System.out.println("1) 添加课程");
            System.out.println("2) 列出课程");
            System.out.println("3) 删除课程");
            System.out.println("0) 返回");
            System.out.print("选择: ");
            String c = scanner.nextLine().trim();
            switch (c) {
                case "1": addCourse(); break;
                case "2": listCourses(); break;
                case "3": deleteCourse(); break;
                case "0": return;
                default: System.out.println("无效选择");
            }
        }
    }

    private void enrollmentMenu() {
        while (true) {
            System.out.println("\n-- 选课与成绩 --");
            System.out.println("1) 学生选课");
            System.out.println("2) 录入/更新成绩");
            System.out.println("3) 查看学生成绩与GPA");
            System.out.println("4) 查看课程成绩统计");
            System.out.println("0) 返回");
            System.out.print("选择: ");
            String c = scanner.nextLine().trim();
            switch (c) {
                case "1": doEnroll(); break;
                case "2": putGrade(); break;
                case "3": showStudentGradesAndGpa(); break;
                case "4": showCourseStats(); break;
                case "0": return;
                default: System.out.println("无效选择");
            }
        }
    }

    private void addStudent() {
        System.out.print("学生姓名: ");
        String name = scanner.nextLine().trim();
        if (name.isEmpty()) { System.out.println("姓名不能为空"); return; }
        Student s = new Student(UUID.randomUUID().toString(), name);
        studentRepo.save(s);
        System.out.println("已添加: " + s);
    }

    private void listStudents() {
        List<Student> all = studentRepo.findAll();
        if (all.isEmpty()) { System.out.println("暂无学生"); return; }
        all.forEach(System.out::println);
    }

    private void deleteStudent() {
        System.out.print("学生ID: ");
        String id = scanner.nextLine().trim();
        boolean ok = studentRepo.deleteById(id);
        System.out.println(ok ? "已删除" : "未找到");
    }

    private void addCourse() {
        System.out.print("课程名称: ");
        String name = scanner.nextLine().trim();
        System.out.print("学分(数字): ");
        String creditsStr = scanner.nextLine().trim();
        try {
            int credits = Integer.parseInt(creditsStr);
            if (credits <= 0) { System.out.println("学分需为正数"); return; }
            Course c = new Course(UUID.randomUUID().toString(), name, credits);
            courseRepo.save(c);
            System.out.println("已添加: " + c);
        } catch (NumberFormatException e) {
            System.out.println("学分无效");
        }
    }

    private void listCourses() {
        List<Course> all = courseRepo.findAll();
        if (all.isEmpty()) { System.out.println("暂无课程"); return; }
        all.forEach(System.out::println);
    }

    private void deleteCourse() {
        System.out.print("课程ID: ");
        String id = scanner.nextLine().trim();
        boolean ok = courseRepo.deleteById(id);
        System.out.println(ok ? "已删除" : "未找到");
    }

    private void doEnroll() {
        System.out.print("学生ID: ");
        String sid = scanner.nextLine().trim();
        System.out.print("课程ID: ");
        String cid = scanner.nextLine().trim();
        boolean ok = gradeService.enrollStudentInCourse(sid, cid);
        System.out.println(ok ? "选课成功" : "选课失败，检查ID是否存在或已选过");
    }

    private void putGrade() {
        System.out.print("学生ID: ");
        String sid = scanner.nextLine().trim();
        System.out.print("课程ID: ");
        String cid = scanner.nextLine().trim();
        System.out.print("成绩(0-100): ");
        String g = scanner.nextLine().trim();
        try {
            double grade = Double.parseDouble(g);
            boolean ok = gradeService.putGrade(sid, cid, grade);
            System.out.println(ok ? "已更新" : "更新失败，检查是否已选课");
        } catch (NumberFormatException e) {
            System.out.println("成绩无效");
        }
    }

    private void showStudentGradesAndGpa() {
        System.out.print("学生ID: ");
        String sid = scanner.nextLine().trim();
        var info = gradeService.getStudentGradesAndGpa(sid);
        if (info == null) { System.out.println("未找到学生或该生无选课"); return; }
        System.out.println("学生: " + info.student());
        info.grades().forEach((course, grade) -> System.out.println(course + " -> " + (grade == null ? "未评分" : grade)));
        System.out.printf("GPA: %.2f\n", info.gpa());
    }

    private void showCourseStats() {
        System.out.print("课程ID: ");
        String cid = scanner.nextLine().trim();
        var stats = gradeService.getCourseStats(cid);
        if (stats == null) { System.out.println("未找到课程或无选课"); return; }
        System.out.println("课程: " + stats.course());
        System.out.printf("人数: %d, 平均分: %.2f, 最高: %.2f, 最低: %.2f\n",
                stats.count(), stats.average(), stats.max(), stats.min());
    }
}
