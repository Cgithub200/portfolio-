﻿// <auto-generated />
using System;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using VertoTest.Context;

#nullable disable

namespace VertoTest.Migrations
{
    [DbContext(typeof(VertoAppDbContext))]
    [Migration("20240909173739_Initial")]
    partial class Initial
    {
        /// <inheritdoc />
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "8.0.8")
                .HasAnnotation("Relational:MaxIdentifierLength", 128);

            SqlServerModelBuilderExtensions.UseIdentityColumns(modelBuilder);

            modelBuilder.Entity("VertoTest.Models.ContentPanels", b =>
                {
                    b.Property<int>("ContentItemID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("ContentItemID"));

                    b.Property<bool>("IsActive")
                        .HasColumnType("bit");

                    b.Property<int>("Order")
                        .HasColumnType("int");

                    b.Property<string>("Title")
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("ContentItemID");

                    b.ToTable("ContentPanels");
                });

            modelBuilder.Entity("VertoTest.Models.Image", b =>
                {
                    b.Property<int>("ImageId")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("ImageId"));

                    b.Property<string>("AltText")
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("DateUploaded")
                        .HasColumnType("datetime2");

                    b.Property<string>("FileName")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<string>("FilePath")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<int?>("StaticContentID")
                        .HasColumnType("int");

                    b.HasKey("ImageId");

                    b.HasIndex("StaticContentID");

                    b.ToTable("Images");
                });

            modelBuilder.Entity("VertoTest.Models.StaticContent", b =>
                {
                    b.Property<int>("ContentId")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("ContentId"));

                    b.Property<string>("Body")
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("DateUpdated")
                        .HasColumnType("datetime2");

                    b.Property<int>("Order")
                        .HasColumnType("int");

                    b.Property<int?>("SliderItemID")
                        .HasColumnType("int");

                    b.Property<string>("StaticClass")
                        .HasColumnType("nvarchar(max)");

                    b.Property<string>("Title")
                        .HasMaxLength(200)
                        .HasColumnType("nvarchar(200)");

                    b.HasKey("ContentId");

                    b.HasIndex("SliderItemID");

                    b.ToTable("StaticContents");
                });

            modelBuilder.Entity("VertoTest.Models.Image", b =>
                {
                    b.HasOne("VertoTest.Models.StaticContent", "staticContent")
                        .WithMany("Images")
                        .HasForeignKey("StaticContentID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("staticContent");
                });

            modelBuilder.Entity("VertoTest.Models.StaticContent", b =>
                {
                    b.HasOne("VertoTest.Models.ContentPanels", "SliderItem")
                        .WithMany("StaticContents")
                        .HasForeignKey("SliderItemID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("SliderItem");
                });

            modelBuilder.Entity("VertoTest.Models.ContentPanels", b =>
                {
                    b.Navigation("StaticContents");
                });

            modelBuilder.Entity("VertoTest.Models.StaticContent", b =>
                {
                    b.Navigation("Images");
                });
#pragma warning restore 612, 618
        }
    }
}
